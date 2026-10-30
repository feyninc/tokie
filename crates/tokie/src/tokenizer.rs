//! A loaded `tokenizer.json`. Encode appends ids. It does not clear them.

use crate::bpe::{Pairs, merge_ids};
use crate::fold::{CharFold, seed as fold_seed};
use crate::wordpiece::VocabTrie;
use crate::cache::{Cache, Prep, ProbeView, mix, pack, pack_bounds, pack_spans};
use crate::cut::Cut;
use crate::norm::Norm;
use crate::unigram::Lattice;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::thread;

/// Whether [`Tokenizer::decode`] writes special tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specials {
    /// Write every id.
    Keep,
    /// Omit ids the file marks special.
    Skip,
}

/// A load, encode, or decode failure.
#[derive(Debug)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

const GPT2: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const VOYAGE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const NEMOTRON: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const KIMI: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct Added {
    content: String,
    id: u32,
    /// ASCII needles search through this. Built once so a line does not
    /// reconstruct the matcher.
    finder: Option<memchr::memmem::Finder<'static>>,
}

/// Added-token search for one normalization flag.
///
/// One ASCII needle stays on `memchr`. A set of needles is one automaton,
/// so a file with hundreds of specials is one pass instead of one pass each.
enum Needles {
    Empty,
    Single(Added),
    Multi {
        ac: aho_corasick::AhoCorasick,
        ids: Vec<u32>,
        /// A byte that occurs in every pattern. The haystack cannot match
        /// when this byte is absent.
        required: u8,
        /// `false` when the patterns share no single byte.
        use_required: bool,
    },
}

enum Kind {
    Bpe {
        /// Vocab id of each raw byte after the GPT-2 byte map. `u32::MAX` is missing.
        byte_ids: [u32; 256],
        pairs: Pairs,
        /// Characters whose byte assembly cannot lose to a neighbour.
        fold: Option<CharFold>,
        /// A pretok piece whose byte-level spelling is one vocab token is that
        /// token. Merges run only when the spelling is absent.
        ignore_merges: bool,
    },
    WordPiece {
        trie: VocabTrie,
        unk: u32,
        prefix: String,
        max_chars: usize,
    },
    Unigram {
        lattice: Lattice,
    },
    /// SentencePiece BPE. Symbols are characters, and a missing character
    /// becomes one `<0xHH>` id per byte.
    Spm {
        chars: FxHashMap<char, u32>,
        bytes: [u32; 256],
        pairs: Pairs,
    },
}

struct Scratch {
    a: String,
    /// Prefix-space and metaspace text for the piece loop. The capacity stays
    /// across calls.
    view: String,
    syms: Vec<u32>,
    piece: Vec<u32>,
    raw_parts: Vec<Part>,
    norm_parts: Vec<Part>,
    /// Byte-level spelling of the current piece, for `ignore_merges`.
    mapped: String,
}

impl Scratch {
    fn new() -> Self {
        Self {
            a: String::new(),
            view: String::new(),
            syms: Vec::new(),
            piece: Vec::new(),
            raw_parts: Vec::new(),
            norm_parts: Vec::new(),
            mapped: String::new(),
        }
    }
}

/// One decoded piece. `at == u32::MAX` means the id is not in the vocab.
#[derive(Clone, Copy)]
struct Slice {
    at: u32,
    len: u32,
}

/// Id to bytes, built at load. `encode` does not read it.
struct Vocab {
    arena: Vec<u8>,
    /// Bytes for this id when it is the first id `decode` writes.
    lead: Vec<Slice>,
    /// Bytes for this id when another id was already written.
    next: Vec<Slice>,
    tokens: Vec<Option<String>>,
    ids: FxHashMap<String, u32>,
    special: Vec<bool>,
    /// Drop one leading space after the slices are joined. Llama's strip does this.
    strip_space: bool,
}

/// Tables a forked tokenizer shares. The cache stays on each [`Tokenizer`].
struct Tables {
    steps: Vec<Step>,
    raw_added: Needles,
    norm_added: Needles,
    /// The strings `raw_added` was built from. `save` writes these, not the automaton.
    raw_rows: Vec<(String, u32)>,
    norm_rows: Vec<(String, u32)>,
    kind: Kind,
    /// Specials the single-sequence template adds around the encoded ids.
    post: Post,
    vocab: Vocab,
    /// Longest-match walk for a long unfolded piece. Empty until the first
    /// such piece, so an English encode never builds it.
    scan: OnceLock<Option<crate::scan::Scan>>,
}

/// Worker tokenizers kept across `batch` calls so their piece caches stay warm.
/// Each `batch` borrows the caller's texts for the scope of that call, so the
/// documents are not copied into `String`s.
struct Pool {
    kids: Vec<Tokenizer>,
}

impl Pool {
    fn new() -> Self {
        Self { kids: Vec::new() }
    }

    fn stop(&mut self) {
        self.kids.clear();
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One tokenizer. The cache and the scratch stay with it. The tables are shared.
///
/// `cut`, `prefix_space`, and `mark` are copied here so a line does not chase
/// the `Arc` before the piece loop. `fork` copies them again.
pub struct Tokenizer {
    cache: Cache,
    scratch: Scratch,
    cut: Cut,
    prefix_space: bool,
    mark: bool,
    /// SHA-256 of the `tokenizer.json` this file was built from. Zeros when that
    /// source was not a json file.
    sha: [u8; 32],
    /// Piece list written by `save`. Encode updates `cache` and leaves this list.
    piece_seed: Arc<Vec<(Vec<u8>, Vec<u32>)>>,
    tables: Arc<Tables>,
    threads: usize,
    pool: Pool,
}

impl Tokenizer {
    /// Read a `.tokie` file or a `tokenizer.json`.
    ///
    /// A file that starts with `TOKI` and is not this format is refused.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Parse a `tokenizer.json`.
    pub fn json(text: &str) -> Result<Self, Error> {
        let mut tok = Self::parse_json(text)?;
        tok.sha = crate::file::sha256(text.as_bytes());
        Ok(tok)
    }

    fn parse_json(text: &str) -> Result<Self, Error> {
        let value: Value = serde_json::from_str(text).map_err(|err| format!("json: {err}"))?;
        let root = value.as_object().ok_or("tokenizer.json is not an object")?;
        let steps = norms_of(root.get("normalizer"))?;
        let (cut, prefix_space, mark) = cut_of(root.get("pre_tokenizer"))?;
        let (raw_added, norm_added, raw_rows, norm_rows) = added_of(root.get("added_tokens"))?;
        let post = post_of(root.get("post_processor"))?;
        let model = root.get("model").and_then(|v| v.as_object()).ok_or("missing model")?;
        let mut kind = kind_of(model)?;
        let vocab = vocab_of(model, root.get("added_tokens"), root.get("decoder"), &kind)?;
        attach_fold(&mut kind, &vocab.tokens);
        let mut tok = Self {
            cache: Cache::new(),
            scratch: Scratch::new(),
            cut,
            prefix_space,
            mark,
            sha: [0; 32],
            piece_seed: Arc::new(Vec::new()),
            tables: Arc::new(Tables {
                steps,
                raw_added,
                norm_added,
                raw_rows,
                norm_rows,
                kind,
                post,
                vocab,
                scan: OnceLock::new(),
            }),
            threads: default_threads(),
            pool: Pool::new(),
        };
        warm_bpe_cache(&mut tok);
        Ok(tok)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() >= 8 && &bytes[..8] == b"TOKIE\x89\r\n" {
            return from_image(crate::file::decode(bytes)?);
        }
        if bytes.len() >= 4 && &bytes[..4] == b"TOKI" {
            return Err(Error::new("old .tkz files are not loaded"));
        }
        let text = std::str::from_utf8(bytes).map_err(|_| "tokenizer file is not utf-8")?;
        Self::json(text)
    }

    /// Write this tokenizer as a `.tokie` file.
    ///
    /// The piece blob is the list this tokenizer was loaded or preloaded with.
    /// Ids learned by `encode` stay in the cache and are not written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let bytes = crate::file::encode(&self.image())?;
        let path = path.as_ref();
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp = Path::new(&tmp_name);
        fs::write(tmp, &bytes).map_err(|err| format!("write {}: {err}", tmp.display()))?;
        let file = fs::File::open(tmp).map_err(|err| format!("open {}: {err}", tmp.display()))?;
        file.sync_all()
            .map_err(|err| format!("sync {}: {err}", tmp.display()))?;
        fs::rename(tmp, path).map_err(|err| format!("rename {}: {err}", path.display()))?;
        Ok(())
    }

    /// Encode `piece` and keep the ids in the piece list `save` writes.
    #[doc(hidden)]
    pub fn preload_piece(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        self.scratch.piece.clear();
        if self.encode_piece(piece, false).is_err() {
            return;
        }
        let encoded = std::mem::take(&mut self.scratch.piece);
        self.cache.insert(piece.as_bytes(), &encoded);
        Arc::make_mut(&mut self.piece_seed).push((piece.as_bytes().to_vec(), encoded));
    }

    fn image(&self) -> crate::file::Image {
        use crate::file::{CharsmapRec, Image, Model, SlotRec, StepRec};
        let tables = &self.tables;
        let model;
        let mut unk = 0u32;
        let mut max_chars = 0u32;
        let mut prefix = String::new();
        let mut unk_score = 0.0;
        let mut pairs = Vec::new();
        let mut byte_ids = None;
        let mut ignore_merges = false;
        let mut spm_bytes = None;
        let mut spm_chars = Vec::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut wp = Vec::new();
        match &tables.kind {
            Kind::Bpe {
                byte_ids: ids,
                pairs: map,
                ignore_merges: skip,
                ..
            } => {
                model = Model::Bpe;
                byte_ids = Some(*ids);
                pairs = map.rows().to_vec();
                ignore_merges = *skip;
            }
            Kind::WordPiece {
                trie,
                unk: unk_id,
                prefix: pre,
                max_chars: max,
            } => {
                model = Model::WordPiece;
                unk = *unk_id;
                max_chars = *max as u32;
                prefix = pre.clone();
                wp = trie.rows().to_vec();
            }
            Kind::Unigram { lattice } => {
                model = Model::Unigram;
                unk = lattice.unk();
                unk_score = lattice.unk_score();
                nodes = lattice.nodes().collect();
                edges = lattice.edges().collect();
            }
            Kind::Spm {
                chars,
                bytes,
                pairs: map,
            } => {
                model = Model::Spm;
                spm_bytes = Some(*bytes);
                spm_chars = chars.iter().map(|(&ch, &id)| (ch as u32, id)).collect();
                pairs = map.rows().to_vec();
            }
        }
        let mut steps = Vec::new();
        let mut literals = Vec::new();
        let mut charsmaps = Vec::new();
        for step in &tables.steps {
            match step {
                Step::Norm(norm) => steps.push(StepRec::Norm(*norm)),
                Step::Compiled { map, blob } => {
                    let index = u16::try_from(charsmaps.len()).expect("at most four charsmaps");
                    let (_, _, bmp) = map.parts();
                    charsmaps.push(CharsmapRec {
                        blob: blob.clone(),
                        bmp: bmp.to_vec(),
                    });
                    steps.push(StepRec::Compiled(index));
                }
                Step::Literal { from, to } => {
                    let index = u16::try_from(literals.len()).expect("literal index fits u16");
                    literals.push((from.clone(), to.clone()));
                    steps.push(StepRec::Literal(index));
                }
            }
        }
        let vocab = &tables.vocab;
        let mut slots = Vec::with_capacity(vocab.lead.len());
        for i in 0..vocab.lead.len() {
            slots.push(SlotRec {
                lead_at: vocab.lead[i].at,
                lead_len: vocab.lead[i].len,
                next_at: vocab.next[i].at,
                next_len: vocab.next[i].len,
                token: vocab.tokens[i].clone(),
                special: vocab.special[i],
            });
        }
        Image {
            sha: self.sha,
            model,
            cut: self.cut,
            prefix_space: self.prefix_space,
            mark: self.mark,
            strip_space: vocab.strip_space,
            ignore_merges,
            unk,
            max_chars,
            vocab_len: slots.len() as u32,
            prefix,
            unk_score,
            steps,
            literals,
            charsmaps,
            arena: vocab.arena.clone(),
            slots,
            pairs,
            byte_ids,
            spm_bytes,
            spm_chars,
            nodes,
            edges,
            added_raw: tables.raw_rows.clone(),
            added_norm: tables.norm_rows.clone(),
            post_prefix: tables.post.prefix.clone(),
            post_suffix: tables.post.suffix.clone(),
            wp,
            pieces: (*self.piece_seed).clone(),
        }
    }

    /// Download a tokenizer for a Hub repo.
    ///
    /// The repo's `.tokie` file is opened, whatever its name. A repo with no
    /// `.tokie` file is parsed from `tokenizer.json`. The file stays in the Hub
    /// cache. A later call with the same repo reads that copy. A build without
    /// the `hub` feature keeps the method and returns an error.
    pub fn pretrained(repo: &str) -> Result<Self, Error> {
        #[cfg(feature = "hub")]
        {
            let api = hf_hub::api::sync::Api::new().map_err(|err| Error::new(format!("hub: {err}")))?;
            let model = api.model(repo.to_string());
            let info = model.info().map_err(|err| Error::new(format!("hub: {err}")))?;
            let mut tokie_files: Vec<&str> = info
                .siblings
                .iter()
                .map(|file| file.rfilename.as_str())
                .filter(|name| name.ends_with(".tokie"))
                .collect();
            tokie_files.sort_unstable();
            return match tokie_files.as_slice() {
                [name] => {
                    let path = model.get(name).map_err(|err| Error::new(format!("hub: {err}")))?;
                    Self::open(path)
                }
                [] => {
                    let path = model
                        .get("tokenizer.json")
                        .map_err(|err| Error::new(format!("hub: {err}")))?;
                    let text = fs::read_to_string(&path)
                        .map_err(|err| format!("read {}: {err}", path.display()))?;
                    Self::json(&text)
                }
                names => Err(Error::new(format!(
                    "hub: {repo} has several .tokie files: {}",
                    names.join(", ")
                ))),
            };
        }
        #[cfg(not(feature = "hub"))]
        {
            let _ = repo;
            Err(Error::new("hub download was left out of this build"))
        }
    }

    /// `(filled slots, capacity, table address)` after the encodes so far.
    #[doc(hidden)]
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        self.cache.occupancy()
    }

    /// Append the ids for `text`. A special token listed in the file is
    /// emitted when it occurs in the text. [`Self::wrap`] applies the template.
    #[inline]
    pub fn encode(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        // No rewrite and no special in the text: one segment. The general
        // path also searches, then builds a part list around misses.
        let plain = self.tables.steps.is_empty() && !self.prefix_space;
        if plain && !contains_added(text, &self.tables.raw_added, &self.tables.norm_added) {
            ids.reserve(text.len() / 4 + 8);
            return self.encode_segment(text, ids).map_err(Error::from);
        }
        self.encode_general(text, ids).map_err(Error::from)
    }

    /// [`Self::encode`], then the single-sequence template.
    ///
    /// A ByteLevel post-processor adds nothing, so those ids match
    /// [`Self::encode`]. BERT uncased is `[CLS]`, the ids, `[SEP]`.
    pub fn wrap(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        let prefix = self.tables.post.prefix.clone();
        let suffix = self.tables.post.suffix.clone();
        let at = ids.len();
        self.encode(text, ids)?;
        if !prefix.is_empty() {
            ids.splice(at..at, prefix);
        }
        ids.extend_from_slice(&suffix);
        Ok(())
    }

    /// Append the text for `ids`. `Keep` writes every id. `Skip` omits specials.
    pub fn decode(&self, ids: &[u32], text: &mut String, specials: Specials) -> Result<(), Error> {
        let vocab = &self.tables.vocab;
        let mut bytes: Vec<u8> = Vec::new();
        let mut emitted = false;
        for &id in ids {
            let index = id as usize;
            let slice = vocab.lead.get(index).ok_or_else(|| Error::new(format!("unknown id {id}")))?;
            if slice.at == u32::MAX {
                return Err(Error::new(format!("unknown id {id}")));
            }
            if specials == Specials::Skip && vocab.special.get(index).copied().unwrap_or(false) {
                continue;
            }
            let slice = if emitted { &vocab.next[index] } else { slice };
            let start = slice.at as usize;
            bytes.extend_from_slice(&vocab.arena[start..start + slice.len as usize]);
            emitted = true;
        }
        let view = if vocab.strip_space && bytes.first() == Some(&b' ') {
            &bytes[1..]
        } else {
            bytes.as_slice()
        };
        text.push_str(&String::from_utf8_lossy(view));
        Ok(())
    }

    /// The vocab string for `id`.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tables.vocab.tokens.get(id as usize)?.as_deref()
    }

    /// The id for a vocab string.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.tables.vocab.ids.get(token).copied()
    }

    /// How many workers [`Self::batch`] uses. `1` stays on the calling thread.
    pub fn threads(&mut self, n: usize) {
        let n = n.max(1);
        if n != self.threads {
            self.pool.stop();
            self.threads = n;
        }
    }

    /// A tokenizer with these tables and a copy of the loaded piece cache.
    ///
    /// Ids learned after load stay on this tokenizer. The copy starts from the
    /// piece list `save` writes.
    pub fn fork(&self) -> Self {
        Self {
            cache: self.cache.duplicate(),
            scratch: Scratch::new(),
            cut: self.cut,
            prefix_space: self.prefix_space,
            mark: self.mark,
            sha: self.sha,
            piece_seed: Arc::clone(&self.piece_seed),
            tables: Arc::clone(&self.tables),
            threads: 1,
            pool: Pool::new(),
        }
    }

    /// Append the ids for each text, in order. `ends[i]` is `ids.len()` after text `i`.
    ///
    /// On error, `ids` and `ends` hold the texts that finished.
    pub fn batch(&mut self, texts: &[&str], ids: &mut Vec<u32>, ends: &mut Vec<usize>) -> Result<(), Error> {
        if texts.is_empty() {
            return Ok(());
        }
        let workers = self.threads.max(1);
        if workers == 1 || texts.len() == 1 {
            return self.batch_here(texts, ids, ends);
        }
        self.ensure_kids(workers);
        let ranges = chunk_ranges(texts.len(), self.pool.kids.len());
        // One token per 3 input bytes over-allocates GPT-2 on OpenWebText
        // (about one per 4.4 bytes) and still covers every BPE corpus we ship.
        // Each worker writes at the front of its region. A spill means that
        // guess was short and that chunk owns its ids instead.
        let mut budgets = Vec::with_capacity(ranges.len());
        for &(start, end) in &ranges {
            let bytes: usize = texts[start..end].iter().map(|text| text.len()).sum();
            let docs = end - start;
            budgets.push(bytes / 3 + docs * 8 + 8);
        }
        let est: usize = budgets.iter().sum();
        let base = ids.len();
        ids.reserve(est);
        let fills = thread::scope(|scope| {
            let spare = &mut ids.spare_capacity_mut()[..est];
            let mut rest = spare;
            let mut regions = Vec::with_capacity(budgets.len());
            for budget in &budgets {
                let (region, tail) = rest.split_at_mut(*budget);
                rest = tail;
                regions.push(region);
            }
            let mut joins = Vec::with_capacity(ranges.len());
            for ((kid, (start, end)), region) in self.pool.kids.iter_mut().zip(ranges).zip(regions) {
                let slice = &texts[start..end];
                joins.push(scope.spawn(move || fill_region(kid, slice, region)));
            }
            joins.into_iter().map(|join| join.join().expect("batch worker")).collect::<Vec<_>>()
        });
        place(ids, ends, base, &budgets, &fills)
    }

    fn batch_here(&mut self, texts: &[&str], ids: &mut Vec<u32>, ends: &mut Vec<usize>) -> Result<(), Error> {
        for text in texts {
            self.encode(text, ids)?;
            ends.push(ids.len());
        }
        Ok(())
    }

    fn ensure_kids(&mut self, n: usize) {
        if self.pool.kids.len() == n {
            return;
        }
        let kids = (0..n).map(|_| self.fork()).collect();
        self.pool.kids = kids;
    }

    fn encode_general(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        ids.reserve(text.len() / 4 + 8);
        let mut parts = std::mem::take(&mut self.scratch.raw_parts);
        split_added(text, &self.tables.raw_added, &mut parts);
        for i in 0..parts.len() {
            match parts[i] {
                Part::Id(id) => ids.push(id),
                Part::Text(start, end) => self.encode_raw(&text[start..end], ids)?,
            }
        }
        self.scratch.raw_parts = parts;
        Ok(())
    }

    fn encode_raw(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        let owned = self.rewrite(text);
        let segment = owned.as_deref().unwrap_or(text);
        let mut parts = std::mem::take(&mut self.scratch.norm_parts);
        split_added(segment, &self.tables.norm_added, &mut parts);
        for i in 0..parts.len() {
            match parts[i] {
                Part::Id(id) => ids.push(id),
                Part::Text(start, end) => self.encode_segment(&segment[start..end], ids)?,
            }
        }
        self.scratch.norm_parts = parts;
        Ok(())
    }

    /// Run the normalizer steps in file order. `None` means every step left
    /// the text alone.
    fn rewrite(&mut self, text: &str) -> Option<String> {
        if self.tables.steps.is_empty() {
            return None;
        }
        let steps = &self.tables.steps;
        let scratch = &mut self.scratch;
        let mut cur = String::new();
        let mut src_is_cur = false;
        let mut changed = false;
        for step in steps {
            let src = if src_is_cur { cur.as_str() } else { text };
            scratch.a.clear();
            let wrote = match step {
                Step::Norm(norm) => norm.write(src, true, &mut scratch.a),
                Step::Compiled { map, .. } => map.write(src, true, &mut scratch.a),
                Step::Literal { from, to } => literal_replace(src, from, to, &mut scratch.a),
            };
            if wrote {
                std::mem::swap(&mut cur, &mut scratch.a);
                src_is_cur = true;
                changed = true;
            }
        }
        changed.then_some(cur)
    }

    #[inline]
    fn encode_segment(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        if self.cache.crc() {
            self.encode_cut::<true>(text, ids)
        } else {
            self.encode_cut::<false>(text, ids)
        }
    }

    #[inline]
    fn encode_cut<const CRC: bool>(&mut self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        let prefix_space = self.prefix_space;
        let mark = self.mark;
        let cut = self.cut;
        let mut buf = std::mem::take(&mut self.scratch.view);
        buf.clear();
        let prefixed = prefix_space && !text.is_empty() && !text.starts_with(' ');
        if prefixed {
            buf.push(' ');
            buf.push_str(text);
        }
        if mark {
            let mut marked = std::mem::take(&mut self.scratch.a);
            marked.clear();
            let src = if prefixed { buf.as_str() } else { text };
            mark_words(src, &mut marked);
            self.scratch.a = std::mem::replace(&mut buf, marked);
        }
        let view = if prefixed || mark { buf.as_str() } else { text };
        let result = match cut {
            Cut::None => {
                if view.is_empty() {
                    Ok(())
                } else {
                let batch = [pretokie::Span {
                    start: 0,
                    end: u32::try_from(view.len()).expect("text is longer than 4 GiB"),
                }];
                let mut prep = [Prep { key: 0, hash: 0 }; 80];
                    let piece = view.as_bytes();
                    let key = pack(piece).unwrap_or(0);
                    let hash = if key == 0 { 0 } else { mix::<CRC>(key) };
                    prep[0] = Prep { key, hash };
                    self.emit_chunk::<CRC>(&prep, &batch, view, ids)
                }
            }
            Cut::Gpt2 => {
                let mut walk = pretokie::Gpt2::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::SmolLM => {
                let mut walk = pretokie::SmolLM::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Cl100k => {
                let mut walk = pretokie::Cl100k::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Olmo => {
                let mut walk = pretokie::Olmo::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Voyage => {
                let mut walk = pretokie::Voyage::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Qwen => {
                let mut walk = pretokie::Qwen::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::O200k => {
                let mut walk = pretokie::O200k::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Nemotron => {
                let mut walk = pretokie::Nemotron::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Kimi => {
                let mut walk = pretokie::Kimi::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::DeepSeek => {
                let mut walk = pretokie::DeepSeek::walk(view);
                self.drive_bounds::<CRC>(|buf, limit| walk.bounds(buf, limit), view, ids)
            }
            Cut::Bert => {
                let mut walk = pretokie::Bert::walk(view);
                self.drive::<CRC>(|out| walk.fill(out), view, ids)
            }
            Cut::T5 => {
                let mut walk = pretokie::T5::walk(view);
                self.drive::<CRC>(|out| walk.fill(out), view, ids)
            }
        };
        self.scratch.view = buf;
        result
    }

    /// Fill a batch, pack and prefetch it, then probe before the next fill.
    /// The rest of a 256-piece pack covers the prefetch, and the probe runs
    /// while those lines are still in cache.
    #[inline(never)]
    fn drive<const CRC: bool>(
        &mut self,
        mut fill: impl FnMut(&mut [pretokie::Span]) -> usize,
        view: &str,
        ids: &mut Vec<u32>,
    ) -> Result<(), String> {
        let bytes = view.as_bytes();
        let mut spans = [std::mem::MaybeUninit::<pretokie::Span>::uninit(); 256];
        let spans = unsafe {
            std::slice::from_raw_parts_mut(spans.as_mut_ptr().cast::<pretokie::Span>(), spans.len())
        };
        let mut prep = [std::mem::MaybeUninit::<Prep>::uninit(); 272];
        loop {
            let n = fill(spans);
            if n == 0 {
                break;
            }
            let ready = unsafe {
                std::slice::from_raw_parts_mut(prep.as_mut_ptr().cast::<Prep>(), n + 16)
            };
            pack_spans::<CRC>(bytes, &spans[..n], ready, self.cache.view());
            self.emit_chunk::<CRC>(ready, &spans[..n], view, ids)?;
            if n < spans.len() {
                break;
            }
        }
        Ok(())
    }

/// Fill boundaries, pack them, then probe. The walker stores offsets
/// only. This pack reads the bytes between those offsets.
    #[inline(never)]
    fn drive_bounds<const CRC: bool>(
        &mut self,
        mut pull: impl FnMut(&mut [u32], usize) -> usize,
        view: &str,
        ids: &mut Vec<u32>,
    ) -> Result<(), String> {
        let bytes = view.as_bytes();
        let mut bounds = [std::mem::MaybeUninit::<u32>::uninit(); 336];
        let bounds = unsafe {
            std::slice::from_raw_parts_mut(bounds.as_mut_ptr().cast::<u32>(), bounds.len())
        };
        let mut prep = [std::mem::MaybeUninit::<Prep>::uninit(); 272];
        loop {
            let n = pull(bounds, 256);
            if n == 0 {
                break;
            }
            let ready = unsafe {
                std::slice::from_raw_parts_mut(prep.as_mut_ptr().cast::<Prep>(), n + 16)
            };
            pack_bounds::<CRC>(bytes, bounds, n, ready, self.cache.view());
            self.emit_bounds::<CRC>(ready, bounds.as_ptr(), n, view, ids)?;
            if n < 256 {
                break;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn emit_bounds<const CRC: bool>(
        &mut self,
        prep: &[Prep],
        bounds: *const u32,
        n: usize,
        text: &str,
        ids: &mut Vec<u32>,
    ) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        const DISTANCE: usize = 16;
        debug_assert!(prep.len() >= n + DISTANCE);
        ids.reserve(4 * n);
        let mut w = ids.len();
        let mut dst = ids.as_mut_ptr();
        let mut table = self.cache.view();
        let prep = prep.as_ptr();
        let mut i = 0;
        while i < n {
            let (done, wrote) = emit_fast(unsafe { prep.add(i) }, n - i, dst, w, table);
            w = wrote;
            i += done;
            if i == n {
                break;
            }
            unsafe { ids.set_len(w) };
            let start = unsafe { *bounds.add(i) } as usize;
            let end = unsafe { *bounds.add(i + 1) } as usize;
            let piece = unsafe { text.get_unchecked(start..end) };
            self.recover::<CRC>(piece, ids)?;
            w = ids.len();
            i += 1;
            if i == n {
                break;
            }
            ids.reserve(4 * (n - i));
            dst = ids.as_mut_ptr();
            table = self.cache.view();
        }
        unsafe { ids.set_len(w) };
        Ok(())
    }

    /// Home-pair hits write straight into `ids`. Anything else goes to
    /// [`Self::recover`].
    #[inline(always)]
    fn emit_chunk<const CRC: bool>(
        &mut self,
        prep: &[Prep],
        batch: &[pretokie::Span],
        text: &str,
        ids: &mut Vec<u32>,
    ) -> Result<(), String> {
        let n = batch.len();
        if n == 0 {
            return Ok(());
        }
        const DISTANCE: usize = 16;
        debug_assert!(prep.len() >= n + DISTANCE);
        ids.reserve(4 * n);
        let mut w = ids.len();
        let mut dst = ids.as_mut_ptr();
        let mut table = self.cache.view();
        let prep = prep.as_ptr();
        for i in 0..DISTANCE.min(n) {
            table.prefetch::<true>(unsafe { (*prep.add(i)).hash });
        }
        for i in 0..n {
            let ahead = unsafe { *prep.add(i + DISTANCE) };
            table.prefetch::<true>(ahead.hash);
            let item = unsafe { *prep.add(i) };
            let (val, ext, found) = table.probe_pair(item.key, item.hash);
            let fast = found & ((val & 0x80) == 0) & (item.key != 0);
            let ab = pack_pair(val);
            unsafe {
                let p = dst.add(w);
                (p as *mut u64).write_unaligned(ab);
                (p.add(2) as *mut u64).write_unaligned(ext);
            }
            w += if fast { (val & 0x7f) as usize } else { 0 };
            if !fast {
                unsafe { ids.set_len(w) };
                let span = unsafe { *batch.as_ptr().add(i) };
                let piece = unsafe { text.get_unchecked(span.start as usize..span.end as usize) };
                self.recover::<CRC>(piece, ids)?;
                w = ids.len();
                ids.reserve(4 * (n - i - 1));
                dst = ids.as_mut_ptr();
                table = self.cache.view();
            }
        }
        unsafe { ids.set_len(w) };
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn recover<const CRC: bool>(&mut self, piece: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        if piece.is_empty() {
            return Ok(());
        }
        if self.cache.probe::<CRC>(piece.as_bytes(), ids) {
            return Ok(());
        }
        self.scratch.piece.clear();
        self.encode_piece(piece, true)?;
        // Keep the buffer. `admit` copies the ids, and the next miss would
        // otherwise allocate a new one from empty. A medium Han piece waits
        // for a second sight so unique pieces do not grow the map.
        self.cache.admit(piece.as_bytes(), &self.scratch.piece);
        ids.extend_from_slice(&self.scratch.piece);
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn encode_piece(&mut self, piece: &str, walk: bool) -> Result<(), String> {
        if matches!(self.tables.kind, Kind::Bpe { ignore_merges: true, .. }) {
            let map = byte_chars();
            self.scratch.mapped.clear();
            for &byte in piece.as_bytes() {
                self.scratch.mapped.push(map[byte as usize]);
            }
            if let Some(&id) = self.tables.vocab.ids.get(&self.scratch.mapped) {
                self.scratch.piece.push(id);
                return Ok(());
            }
        }
        let mut seeded = false;
        if walk && matches!(self.tables.kind, Kind::Bpe { .. }) {
            // The walk does not read the folded symbols. Once it exists, a
            // Thai piece of 16 bytes or more skips the fold, and so does a
            // medium Han piece when the Han fold is too sparse to shrink it.
            // This does not build the walk. Kana stays on the fold.
            let raw = piece.as_bytes();
            // GPT-2 stays under 65536 tokens, so the heap is the usual path.
            // A Thai piece in that range is still a stack merge of raw bytes,
            // and nothing else in the text is long enough to build the walk.
            let mut ready = matches!(self.tables.scan.get(), Some(Some(_)));
            if !ready && raw.len() >= 16 && crate::scan::thai_piece(raw) {
                self.ensure_scan_for(true);
                ready = matches!(self.tables.scan.get(), Some(Some(_)));
            }
            let sparse = match &self.tables.kind {
                Kind::Bpe { fold: Some(fold), .. } => fold.sparse_han(),
                _ => false,
            };
            let gate = ready
                && (crate::scan::worth_direct(raw, true)
                    || crate::scan::worth_sparse_han(raw, true, sparse));
            if gate {
                let took = match (self.tables.scan.get(), &self.tables.kind) {
                    (Some(Some(scan)), Kind::Bpe { pairs, .. }) => {
                        if raw.len() <= 64 {
                            scan.try_short(pairs, raw, &mut self.scratch.piece)
                        } else {
                            scan.try_encode(pairs, raw, &mut self.scratch.piece)
                        }
                    }
                    _ => false,
                };
                if took {
                    return Ok(());
                }
            }
        }
        if walk
            && let Kind::Bpe { byte_ids, fold, .. } = &self.tables.kind
        {
            self.scratch.syms.clear();
            if let Some(fold) = fold {
                fold_seed(fold, byte_ids, piece.as_bytes(), |id| self.scratch.syms.push(id));
            } else {
                for &byte in piece.as_bytes() {
                    let id = byte_ids[byte as usize];
                    if id == u32::MAX {
                        return Err(format!("no vocab id for byte {byte:#04x}"));
                    }
                    self.scratch.syms.push(id);
                }
            }
            seeded = true;
            let symbols = self.scratch.syms.len();
            let bytes = piece.len();
            // Build only for a long unfolded piece, so English and OpenWebText
            // never allocate the automaton. Once it exists, pieces past the
            // stack merge reuse it, including a sparse model's piece that
            // folded and still has too many symbols for the stack. A dense
            // Han fold keeps the queue.
            let ready = matches!(self.tables.scan.get(), Some(Some(_)));
            let sparse = fold.as_ref().is_some_and(|fold| fold.sparse_han());
            // GPT-2 is under 65536 tokens, so a long Japanese piece never
            // builds the walk and stays on the heap.
            let small = self.tables.vocab.tokens.len() <= (1 << 16);
            if crate::scan::worth_walking(symbols, bytes)
                || (small && crate::scan::worth_small_long(symbols, bytes, sparse))
                || (ready && crate::scan::worth_reusing(symbols, bytes))
                || crate::scan::worth_folded_walk(symbols, bytes, ready, sparse)
            {
                if !ready {
                    self.ensure_scan_for(small);
                }
                let took = match (self.tables.scan.get(), &self.tables.kind) {
                    (Some(Some(scan)), Kind::Bpe { pairs, .. }) => {
                        scan.try_encode(pairs, piece.as_bytes(), &mut self.scratch.piece)
                    }
                    _ => false,
                };
                if took {
                    return Ok(());
                }
            }
        }
        let kind = &self.tables.kind;
        let scratch = &mut self.scratch;
        match kind {
            Kind::Bpe { byte_ids, pairs, fold, .. } => {
                if !seeded {
                    scratch.syms.clear();
                    if let Some(fold) = fold {
                        fold_seed(fold, byte_ids, piece.as_bytes(), |id| scratch.syms.push(id));
                    } else {
                        for &byte in piece.as_bytes() {
                            let id = byte_ids[byte as usize];
                            if id == u32::MAX {
                                return Err(format!("no vocab id for byte {byte:#04x}"));
                            }
                            scratch.syms.push(id);
                        }
                    }
                }
                merge_ids(&mut scratch.syms, pairs);
                scratch.piece.extend_from_slice(&scratch.syms);
            }
            Kind::WordPiece {
                trie,
                unk,
                prefix,
                max_chars,
            } => {
                trie.encode(piece, *unk, *max_chars, prefix, &mut scratch.piece);
            }
            Kind::Unigram { lattice } => {
                lattice.segment(piece.as_bytes(), &mut scratch.piece);
            }
            Kind::Spm { chars, bytes, pairs } => {
                scratch.syms.clear();
                for ch in piece.chars() {
                    if let Some(&id) = chars.get(&ch) {
                        scratch.syms.push(id);
                        continue;
                    }
                    let mut buf = [0u8; 4];
                    for &byte in ch.encode_utf8(&mut buf).as_bytes() {
                        scratch.syms.push(bytes[byte as usize]);
                    }
                }
                merge_ids(&mut scratch.syms, pairs);
                scratch.piece.extend_from_slice(&scratch.syms);
            }
        }
        Ok(())
    }

    /// Build the long-piece walk once. A vocab the heap still wins on stores
    /// `None`, so later long pieces do not try again.
    ///
    /// `allow_small` builds it for a vocab at or below 65536 tokens. That is
    /// the Thai path and a long Japanese piece. English never asks.
    fn ensure_scan_for(&self, allow_small: bool) {
        if self.tables.scan.get().is_some() {
            return;
        }
        let built = match &self.tables.kind {
            Kind::Bpe { pairs, byte_ids, .. }
                if allow_small || self.tables.vocab.tokens.len() > (1 << 16) =>
            {
                crate::scan::Scan::build(&self.tables.vocab.tokens, pairs, byte_ids)
            }
            _ => None,
        };
        let _ = self.tables.scan.set(built);
    }
}

#[derive(Clone, Copy)]
enum Part {
    Text(usize, usize),
    Id(u32),
}

fn contains_added(text: &str, raw: &Needles, norm: &Needles) -> bool {
    needle_hits(raw, text) || needle_hits(norm, text)
}

fn needle_hits(needles: &Needles, text: &str) -> bool {
    match needles {
        Needles::Empty => false,
        Needles::Single(token) => find_token(text, token).is_some(),
        Needles::Multi {
            ac,
            required,
            use_required,
            ..
        } => {
            if *use_required && memchr::memchr(*required, text.as_bytes()).is_none() {
                return false;
            }
            ac.find(text.as_bytes()).is_some()
        }
    }
}

/// Home-pair hits for one batch. The output pointer is not a `Vec`, so the
/// loop keeps the table mask and the cursor in registers instead of
/// recounting a reservation. Returns how many pieces were consumed and the
/// new output length. A miss, a spill, or an empty key stops the batch.
#[inline(never)]
fn emit_fast(
    mut prep: *const Prep,
    n: usize,
    dst: *mut u32,
    mut w: usize,
    table: ProbeView,
) -> (usize, usize) {
    const DISTANCE: usize = 16;
    for i in 0..DISTANCE.min(n) {
        table.prefetch::<true>(unsafe { (*prep.add(i)).hash });
    }
    let end = unsafe { prep.add(n) };
    let mut done = 0;
    while prep < end {
        let ahead = unsafe { *prep.add(DISTANCE) };
        table.prefetch::<true>(ahead.hash);
        let item = unsafe { *prep };
        let (val, ext, found) = table.probe_pair(item.key, item.hash);
        let fast = found & ((val & 0x80) == 0) & (item.key != 0);
        let ab = pack_pair(val);
        unsafe {
            let p = dst.add(w);
            (p as *mut u64).write_unaligned(ab);
            (p.add(2) as *mut u64).write_unaligned(ext);
        }
        if !fast {
            return (done, w);
        }
        w += (val & 0x7f) as usize;
        prep = unsafe { prep.add(1) };
        done += 1;
    }
    (n, w)
}

/// Pack token lanes 1 and 2 into one store. The shift is written out so
/// the probe loop does not reload a 64-bit mask for every piece.
#[inline(always)]
fn pack_pair(val: u64) -> u64 {
    let tok1 = u64::from(((val >> 8) as u32) & 0x00ff_ffff);
    let mut tok2 = u64::from((val >> 32) as u32);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "shl {tok2}, 32",
            "or {tok2}, {tok1}",
            tok1 = in(reg) tok1,
            tok2 = inout(reg) tok2,
            options(pure, nomem, nostack),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        tok2 = (tok2 << 32) | tok1;
    }
    tok2
}

fn split_added(text: &str, needles: &Needles, out: &mut Vec<Part>) {
    out.clear();
    match needles {
        Needles::Empty => {
            if !text.is_empty() {
                out.push(Part::Text(0, text.len()));
            }
        }
        Needles::Single(token) => {
            let mut at = 0;
            while at < text.len() {
                let Some(pos) = find_token(&text[at..], token) else {
                    out.push(Part::Text(at, text.len()));
                    return;
                };
                let start = at + pos;
                let end = start + token.content.len();
                if start > at {
                    out.push(Part::Text(at, start));
                }
                out.push(Part::Id(token.id));
                at = end;
            }
        }
        Needles::Multi {
            ac,
            ids,
            required,
            use_required,
        } => {
            let bytes = text.as_bytes();
            if *use_required && memchr::memchr(*required, bytes).is_none() {
                if !text.is_empty() {
                    out.push(Part::Text(0, text.len()));
                }
                return;
            }
            let mut at = 0;
            while at < bytes.len() {
                let Some(hit) = ac.find(&bytes[at..]) else {
                    out.push(Part::Text(at, text.len()));
                    return;
                };
                let start = at + hit.start();
                let end = at + hit.end();
                debug_assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
                if start > at {
                    out.push(Part::Text(at, start));
                }
                out.push(Part::Id(ids[hit.pattern().as_usize()]));
                at = end;
            }
        }
    }
}

/// Byte search for an ASCII special. Those bytes never sit inside a
/// multibyte character, so the match is a char boundary. Anything else uses
/// the string search.
fn find_token(haystack: &str, token: &Added) -> Option<usize> {
    if let Some(finder) = &token.finder {
        finder.find(haystack.as_bytes())
    } else {
        haystack.find(&token.content)
    }
}

fn norms_of(value: Option<&Value>) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    if let Some(value) = value {
        push_norm(value, &mut steps)?;
    }
    Ok(steps)
}

fn push_norm(value: &Value, steps: &mut Vec<Step>) -> Result<(), String> {
    if value.is_null() {
        return Ok(());
    }
    let obj = value.as_object().ok_or("normalizer is not an object")?;
    let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        "Sequence" => {
            let kids = obj
                .get("normalizers")
                .and_then(|v| v.as_array())
                .ok_or("sequence normalizer has no normalizers")?;
            for kid in kids {
                push_norm(kid, steps)?;
            }
        }
        "NFC" => steps.push(Step::Norm(Norm::Nfc)),
        "NFD" => steps.push(Step::Norm(Norm::Nfd)),
        "NFKC" => steps.push(Step::Norm(Norm::Nfkc)),
        "NFKD" => steps.push(Step::Norm(Norm::Nfkd)),
        "Nmt" => steps.push(Step::Norm(Norm::Nmt)),
        "Lowercase" => steps.push(Step::Norm(Norm::Lower)),
        "Strip" => steps.push(Step::Norm(Norm::Strip)),
        "StripAccents" => steps.push(Step::Norm(Norm::StripAccents)),
        "Prepend" => steps.push(Step::Norm(Norm::Prepend)),
        "Replace" => {
            let from = literal_pattern(obj.get("pattern").ok_or("replace has no pattern")?)?;
            let to = obj
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("replace has no content")?
                .to_string();
            if from.is_empty() {
                return Err("replace pattern is empty".to_string());
            }
            if from == " " && to == "▁" {
                steps.push(Step::Norm(Norm::ReplaceSpace));
            } else {
                steps.push(Step::Literal { from, to });
            }
        }
        "Precompiled" => {
            let blob = obj
                .get("precompiled_charsmap")
                .and_then(|v| v.as_str())
                .ok_or("precompiled normalizer has no charsmap")?;
            let bytes = decode_base64(blob)?;
            let map = normalie::Precompiled::parse(&bytes)
                .map_err(|err| format!("precompiled charsmap: {err}"))?;
            steps.push(Step::Compiled { map, blob: bytes });
        }
        "BertNormalizer" => {
            let lowercase = obj.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(false);
            if lowercase {
                steps.push(Step::Norm(Norm::BertUncased));
            } else {
                steps.push(Step::Norm(Norm::BertCased));
                if obj.get("strip_accents").and_then(|v| v.as_bool()) == Some(true) {
                    steps.push(Step::Norm(Norm::StripAccents));
                }
            }
        }
        other => return Err(format!("unsupported normalizer {other}")),
    }
    Ok(())
}

enum Step {
    Norm(Norm),
    Compiled { map: normalie::Precompiled, blob: Vec<u8> },
    Literal { from: String, to: String },
}

fn literal_pattern(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Object(map) => map
            .get("String")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| format!("replace pattern is not a literal: {value}")),
        _ => Err(format!("replace pattern is not a literal: {value}")),
    }
}

fn literal_replace(text: &str, from: &str, to: &str, out: &mut String) -> bool {
    let Some(hit) = text.find(from) else {
        return false;
    };
    out.push_str(&text[..hit]);
    out.push_str(to);
    let mut at = hit + from.len();
    while let Some(next) = text[at..].find(from) {
        let end = at + next;
        out.push_str(&text[at..end]);
        out.push_str(to);
        at = end + from.len();
    }
    out.push_str(&text[at..]);
    true
}

struct PreState {
    patterns: Vec<String>,
    digits: bool,
    bert: bool,
    byte_regex: bool,
    prefix_space: bool,
    whitespace: bool,
    mark: bool,
}

fn cut_of(value: Option<&Value>) -> Result<(Cut, bool, bool), String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok((Cut::None, false, false));
    };
    let mut state = PreState {
        patterns: Vec::new(),
        digits: false,
        bert: false,
        byte_regex: false,
        prefix_space: false,
        whitespace: false,
        mark: false,
    };
    walk_pre(value, &mut state)?;
    let cut = if state.bert {
        Cut::Bert
    } else if state.digits && state.byte_regex {
        Cut::SmolLM
    } else if state.patterns.iter().any(|pattern| pattern == DEEPSEEK) {
        Cut::DeepSeek
    } else if state.patterns.iter().any(|pattern| pattern == O200K) {
        Cut::O200k
    } else if state.patterns.iter().any(|pattern| pattern == NEMOTRON) {
        Cut::Nemotron
    } else if state.patterns.iter().any(|pattern| pattern == KIMI) {
        Cut::Kimi
    } else if state.patterns.iter().any(|pattern| pattern == VOYAGE) {
        Cut::Voyage
    } else if state.patterns.iter().any(|pattern| pattern == QWEN) {
        Cut::Qwen
    } else if state.patterns.iter().any(|pattern| pattern == CL100K) {
        Cut::Cl100k
    } else if state.patterns.iter().any(|pattern| pattern == GPT2) || state.byte_regex {
        Cut::Gpt2
    } else if state.whitespace && state.patterns.is_empty() {
        Cut::T5
    } else {
        return Err(format!("unsupported pretokenizer patterns: {:?}", state.patterns));
    };
    Ok((cut, state.prefix_space, state.mark))
}

fn walk_pre(value: &Value, state: &mut PreState) -> Result<(), String> {
    let obj = value.as_object().ok_or("pre_tokenizer is not an object")?;
    let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        "Sequence" => {
            let kids = obj
                .get("pretokenizers")
                .and_then(|v| v.as_array())
                .ok_or("sequence pretokenizer has no pretokenizers")?;
            for kid in kids {
                walk_pre(kid, state)?;
            }
        }
        "ByteLevel" => {
            if obj.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true) {
                state.byte_regex = true;
            }
            if obj.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false) {
                state.prefix_space = true;
            }
        }
        "Digits" => state.digits = true,
        "BertPreTokenizer" => state.bert = true,
        "WhitespaceSplit" => state.whitespace = true,
        "Metaspace" => {
            let replacement = obj.get("replacement").and_then(|v| v.as_str()).unwrap_or("▁");
            if replacement != "▁" {
                return Err(format!("unsupported metaspace replacement {replacement:?}"));
            }
            state.mark = match obj.get("prepend_scheme").and_then(|v| v.as_str()) {
                Some("always") => true,
                Some("never") => false,
                Some(other) => return Err(format!("unsupported metaspace prepend {other}")),
                None => obj.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(true),
            };
        }
        "Split" => {
            let pattern = obj.get("pattern").ok_or("split has no pattern")?;
            state.patterns.push(canon_pattern(&regex_of(pattern)?));
        }
        other => return Err(format!("unsupported pretokenizer {other}")),
    }
    Ok(())
}

fn canon_pattern(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    for ch in pattern.chars() {
        match ch {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out
}

fn regex_of(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Object(map) => map
            .get("Regex")
            .or_else(|| map.get("String"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| format!("pattern has no regex: {value}")),
        _ => Err(format!("pattern is not a regex: {value}")),
    }
}

struct Post {
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

/// Single-sequence template. Anything other than `TemplateProcessing` adds
/// no ids. Specials before the sequence are the prefix. Specials after it
/// are the suffix.
fn post_of(value: Option<&Value>) -> Result<Post, String> {
    let empty = Post {
        prefix: Vec::new(),
        suffix: Vec::new(),
    };
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(empty);
    };
    let obj = value.as_object().ok_or("post_processor is not an object")?;
    if obj.get("type").and_then(|v| v.as_str()) != Some("TemplateProcessing") {
        return Ok(empty);
    }
    let specials = obj
        .get("special_tokens")
        .and_then(|v| v.as_object())
        .ok_or("template has no special_tokens")?;
    let single = obj
        .get("single")
        .and_then(|v| v.as_array())
        .ok_or("template has no single sequence")?;
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut seen_seq = false;
    for item in single {
        let item = item.as_object().ok_or("template piece is not an object")?;
        if item.contains_key("Sequence") {
            if seen_seq {
                return Err("single template has two sequences".to_string());
            }
            seen_seq = true;
            continue;
        }
        let name = item
            .get("SpecialToken")
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .ok_or("template piece is not a token or a sequence")?;
        let ids = specials
            .get(name)
            .and_then(|v| v.get("ids"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("template token {name} has no ids"))?;
        let dest = if seen_seq { &mut suffix } else { &mut prefix };
        for id in ids {
            let id = id.as_u64().ok_or_else(|| format!("template id for {name} is not an integer"))?;
            dest.push(u32::try_from(id).map_err(|_| format!("template id {id} does not fit in u32"))?);
        }
    }
    Ok(Post { prefix, suffix })
}

fn from_image(image: crate::file::Image) -> Result<Tokenizer, Error> {
    use crate::file::{Model, StepRec};
    let mut kind = match image.model {
        Model::Bpe => Kind::Bpe {
            byte_ids: image.byte_ids.ok_or("bpe file has no byte ids")?,
            pairs: Pairs::from_list(&image.pairs),
            fold: None,
            ignore_merges: image.ignore_merges,
        },
        Model::WordPiece => Kind::WordPiece {
            trie: VocabTrie::from_rows(image.wp.clone()),
            unk: image.unk,
            prefix: image.prefix.clone(),
            max_chars: image.max_chars as usize,
        },
        Model::Unigram => {
            check_lattice(&image.nodes, &image.edges)?;
            Kind::Unigram {
                lattice: Lattice::from_parts(image.nodes.clone(), image.edges.clone(), image.unk, image.unk_score),
            }
        }
        Model::Spm => Kind::Spm {
            chars: char_map(&image.spm_chars)?,
            bytes: image.spm_bytes.ok_or("sentencepiece file has no byte tokens")?,
            pairs: Pairs::from_list(&image.pairs),
        },
    };
    let mut steps = Vec::with_capacity(image.steps.len());
    for step in &image.steps {
        steps.push(match step {
            StepRec::Norm(norm) => Step::Norm(*norm),
            StepRec::Compiled(index) => {
                let rec = image
                    .charsmaps
                    .get(*index as usize)
                    .ok_or("compiled step has no charsmap")?;
                let map = normalie::Precompiled::from_stored(&rec.blob, rec.bmp.clone())
                    .map_err(|err| format!("precompiled charsmap: {err}"))?;
                Step::Compiled {
                    map,
                    blob: rec.blob.clone(),
                }
            }
            StepRec::Literal(index) => {
                let (from, to) = image
                    .literals
                    .get(*index as usize)
                    .ok_or("literal step has no replacement")?;
                Step::Literal {
                    from: from.clone(),
                    to: to.clone(),
                }
            }
        });
    }
    let vocab = vocab_from_slots(&image)?;
    attach_fold(&mut kind, &vocab.tokens);
    let raw_added = needles_of(image.added_raw.clone())?;
    let norm_added = needles_of(image.added_norm.clone())?;
    let mut cache = Cache::new();
    for (piece, ids) in &image.pieces {
        cache.insert(piece, ids);
    }
    let mut tok = Tokenizer {
        cache,
        scratch: Scratch::new(),
        cut: image.cut,
        prefix_space: image.prefix_space,
        mark: image.mark,
        sha: image.sha,
        piece_seed: Arc::new(image.pieces),
        tables: Arc::new(Tables {
            steps,
            raw_added,
            norm_added,
            raw_rows: image.added_raw,
            norm_rows: image.added_norm,
            kind,
            post: Post {
                prefix: image.post_prefix,
                suffix: image.post_suffix,
            },
            vocab,
            scan: OnceLock::new(),
        }),
        threads: default_threads(),
        pool: Pool::new(),
    };
    warm_bpe_cache(&mut tok);
    Ok(tok)
}

fn attach_fold(kind: &mut Kind, tokens: &[Option<String>]) {
    let Kind::Bpe { fold: None, .. } = kind else {
        return;
    };
    let map = unicode_to_byte();
    let raws: Vec<(u32, Vec<u8>)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(id, token)| {
            let token = token.as_ref()?;
            Some((id as u32, byte_token_bytes(token, &map)?))
        })
        .collect();
    let built = match kind {
        Kind::Bpe { byte_ids, pairs, .. } => {
            CharFold::build(byte_ids, pairs, raws.iter().map(|(id, raw)| (*id, raw.as_slice())))
        }
        _ => return,
    };
    if let Kind::Bpe { fold, .. } = kind {
        *fold = built;
    }
}

/// Raw bytes of a byte-level vocab token. `None` when a character is outside
/// the GPT-2 byte alphabet, which is how special tokens are spelled.
fn byte_token_bytes(token: &str, map: &FxHashMap<char, u8>) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(token.chars().count());
    for ch in token.chars() {
        out.push(*map.get(&ch)?);
    }
    Some(out)
}

/// How many lowest-id vocab tokens to merge once at load. A 10 KiB slice is
/// mostly these tokens the first time it is seen, and the piece cache
/// otherwise starts empty.
const SEED_MAX: usize = 1 << 16;

fn warm_bpe_cache(tok: &mut Tokenizer) {
    if !matches!(tok.tables.kind, Kind::Bpe { .. }) {
        return;
    }
    let map = unicode_to_byte();
    let n = tok.tables.vocab.tokens.len().min(SEED_MAX);
    for id in 0..n {
        let Some(token) = tok.tables.vocab.tokens[id].clone() else {
            continue;
        };
        let Some(raw) = byte_token_bytes(&token, &map) else {
            continue;
        };
        let Ok(text) = std::str::from_utf8(&raw) else {
            continue;
        };
        tok.scratch.piece.clear();
        if tok.encode_piece(text, false).is_err() {
            continue;
        }
        let ids = std::mem::take(&mut tok.scratch.piece);
        tok.cache.insert(&raw, &ids);
    }
}

fn char_map(rows: &[(u32, u32)]) -> Result<FxHashMap<char, u32>, String> {
    let mut map = FxHashMap::default();
    for &(scalar, id) in rows {
        let ch = char::from_u32(scalar).ok_or("sentencepiece char is not a scalar")?;
        map.insert(ch, id);
    }
    Ok(map)
}

fn check_lattice(nodes: &[(u32, u16, u32, f64)], edges: &[(u8, u32)]) -> Result<(), String> {
    for &(edge_at, edge_n, _, _) in nodes {
        let end = edge_at as usize + edge_n as usize;
        if end > edges.len() {
            return Err("unigram node edges run past the edge list".to_string());
        }
    }
    for &(_, to) in edges {
        if to as usize >= nodes.len() {
            return Err("unigram edge leaves the lattice".to_string());
        }
    }
    Ok(())
}

fn vocab_from_slots(image: &crate::file::Image) -> Result<Vocab, String> {
    if image.slots.len() != image.vocab_len as usize {
        return Err("slot count does not match the vocab length".to_string());
    }
    let n = image.slots.len();
    let mut vocab = Vocab {
        arena: image.arena.clone(),
        lead: Vec::with_capacity(n),
        next: Vec::with_capacity(n),
        tokens: Vec::with_capacity(n),
        ids: FxHashMap::with_capacity_and_hasher(n, Default::default()),
        special: Vec::with_capacity(n),
        strip_space: image.strip_space,
    };
    for (id, slot) in image.slots.iter().enumerate() {
        let lead = Slice {
            at: slot.lead_at,
            len: slot.lead_len,
        };
        let next = Slice {
            at: slot.next_at,
            len: slot.next_len,
        };
        check_slice(vocab.arena.len(), lead)?;
        check_slice(vocab.arena.len(), next)?;
        vocab.lead.push(lead);
        vocab.next.push(next);
        vocab.special.push(slot.special);
        if let Some(token) = &slot.token {
            vocab.ids.insert(token.clone(), id as u32);
            vocab.tokens.push(Some(token.clone()));
        } else {
            vocab.tokens.push(None);
        }
    }
    Ok(vocab)
}

fn check_slice(arena_len: usize, slice: Slice) -> Result<(), String> {
    if slice.at == u32::MAX {
        return Ok(());
    }
    let end = (slice.at as usize).saturating_add(slice.len as usize);
    if end > arena_len {
        return Err("decode slice runs past the arena".to_string());
    }
    Ok(())
}

fn added_of(
    value: Option<&Value>,
) -> Result<(Needles, Needles, Vec<(String, u32)>, Vec<(String, u32)>), String> {
    let Some(value) = value else {
        return Ok((Needles::Empty, Needles::Empty, Vec::new(), Vec::new()));
    };
    if value.is_null() {
        return Ok((Needles::Empty, Needles::Empty, Vec::new(), Vec::new()));
    }
    let list = value.as_array().ok_or("added_tokens is not a list")?;
    let mut raw = Vec::new();
    let mut norm = Vec::new();
    for token in list {
        let obj = token.as_object().ok_or("added token is not an object")?;
        let content = obj
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("added token has no content")?
            .to_string();
        if content.is_empty() {
            continue;
        }
        let id = obj
            .get("id")
            .and_then(|v| v.as_u64())
            .ok_or("added token has no id")?;
        let id = u32::try_from(id).map_err(|_| format!("added token id {id} does not fit u32"))?;
        let normalized = obj.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false);
        if normalized {
            norm.push((content, id));
        } else {
            raw.push((content, id));
        }
    }
    let raw_rows = raw.clone();
    let norm_rows = norm.clone();
    Ok((needles_of(raw)?, needles_of(norm)?, raw_rows, norm_rows))
}

fn needles_of(tokens: Vec<(String, u32)>) -> Result<Needles, String> {
    if tokens.is_empty() {
        return Ok(Needles::Empty);
    }
    if tokens.len() == 1 {
        let (content, id) = tokens.into_iter().next().unwrap();
        let finder = if content.bytes().all(|byte| byte < 128) {
            Some(memchr::memmem::Finder::new(content.as_bytes()).into_owned())
        } else {
            None
        };
        return Ok(Needles::Single(Added { content, id, finder }));
    }
    let mut ids = Vec::with_capacity(tokens.len());
    let mut patterns = Vec::with_capacity(tokens.len());
    for (content, id) in tokens {
        patterns.push(content);
        ids.push(id);
    }
    let ac = aho_corasick::AhoCorasick::builder()
        .match_kind(aho_corasick::MatchKind::LeftmostLongest)
        .build(&patterns)
        .map_err(|err| format!("added tokens: {err}"))?;
    let (required, use_required) = shared_byte(&patterns);
    Ok(Needles::Multi {
        ac,
        ids,
        required,
        use_required,
    })
}

/// A byte shared by every pattern, if there is one.
fn shared_byte(patterns: &[String]) -> (u8, bool) {
    let Some(first) = patterns.first() else {
        return (0, false);
    };
    for &byte in first.as_bytes() {
        if patterns.iter().all(|pattern| pattern.as_bytes().contains(&byte)) {
            return (byte, true);
        }
    }
    (0, false)
}

fn kind_of(model: &serde_json::Map<String, Value>) -> Result<Kind, String> {
    let byte_fallback = model.get("byte_fallback").and_then(|v| v.as_bool()) == Some(true);
    let merges = model.get("merges").and_then(|v| v.as_array());
    if let Some(merges) = merges.filter(|list| !list.is_empty()) {
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("model has no vocab")?;
        if byte_fallback {
            return spm_of(model, vocab_obj, merges);
        }
        let ignore_merges = model.get("ignore_merges").and_then(|v| v.as_bool()) == Some(true);
        return bpe_of(vocab_obj, merges, ignore_merges);
    }
    if byte_fallback {
        return Err("byte_fallback without merges is not loaded yet".to_string());
    }
    if let Some(list) = model.get("vocab").and_then(|v| v.as_array()) {
        return unigram_of(model, list);
    }
    let vocab_obj = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or("model has no vocab")?;
    wordpiece_of(model, vocab_obj)
}

fn unigram_of(model: &serde_json::Map<String, Value>, list: &[Value]) -> Result<Kind, String> {
    let mut pieces = Vec::with_capacity(list.len());
    for item in list {
        let pair = item.as_array().ok_or("unigram vocab entry is not a pair")?;
        let token = pair
            .first()
            .and_then(|v| v.as_str())
            .ok_or("unigram piece is not a string")?;
        let score = pair
            .get(1)
            .and_then(|v| v.as_f64())
            .ok_or("unigram piece has no score")?;
        pieces.push((token.to_string(), score));
    }
    let unk = model
        .get("unk_id")
        .and_then(|v| v.as_u64())
        .ok_or("unigram has no unk_id")?;
    let unk = u32::try_from(unk).map_err(|_| format!("unk_id {unk} does not fit u32"))?;
    if pieces.is_empty() || unk as usize >= pieces.len() {
        return Err("unk_id is outside the unigram vocab".to_string());
    }
    Ok(Kind::Unigram {
        lattice: Lattice::build(pieces, unk),
    })
}

fn bpe_of(
    vocab_obj: &serde_json::Map<String, Value>,
    merges: &[Value],
    ignore_merges: bool,
) -> Result<Kind, String> {
    let mut vocab: FxHashMap<&str, u32> = FxHashMap::with_capacity_and_hasher(vocab_obj.len(), Default::default());
    let mut singles: FxHashMap<char, u32> = FxHashMap::default();
    for (token, id) in vocab_obj {
        let id = id_of(id)?;
        vocab.insert(token.as_str(), id);
        let mut chars = token.chars();
        if let (Some(ch), None) = (chars.next(), chars.next()) {
            singles.insert(ch, id);
        }
    }
    let mut byte_ids = [u32::MAX; 256];
    for (byte, ch) in bytes_to_unicode().into_iter().enumerate() {
        if let Some(id) = singles.get(&ch) {
            byte_ids[byte] = *id;
        }
    }
    let mut rows = Vec::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let (left, right) = merge_sides(merge)?;
        let left_id = *vocab.get(left).ok_or_else(|| format!("merge left {left:?} is not in the vocab"))?;
        let right_id = *vocab.get(right).ok_or_else(|| format!("merge right {right:?} is not in the vocab"))?;
        let mut joined = String::with_capacity(left.len() + right.len());
        joined.push_str(left);
        joined.push_str(right);
        let id = *vocab.get(joined.as_str()).ok_or_else(|| format!("merge result {joined:?} is not in the vocab"))?;
        let rank = u32::try_from(rank).map_err(|_| "too many merges".to_string())?;
        rows.push((left_id, right_id, rank, id));
    }
    Ok(Kind::Bpe {
        byte_ids,
        pairs: Pairs::from_list(&rows),
        fold: None,
        ignore_merges,
    })
}

fn spm_of(
    model: &serde_json::Map<String, Value>,
    vocab_obj: &serde_json::Map<String, Value>,
    merges: &[Value],
) -> Result<Kind, String> {
    let prefix = model
        .get("continuing_subword_prefix")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let suffix = model
        .get("end_of_word_suffix")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err("affixed byte_fallback bpe is not loaded yet".to_string());
    }
    let mut vocab: FxHashMap<&str, u32> =
        FxHashMap::with_capacity_and_hasher(vocab_obj.len(), Default::default());
    let mut chars: FxHashMap<char, u32> = FxHashMap::default();
    let mut bytes = [u32::MAX; 256];
    for (token, id) in vocab_obj {
        let id = id_of(id)?;
        vocab.insert(token.as_str(), id);
        let mut iter = token.chars();
        if let (Some(ch), None) = (iter.next(), iter.next()) {
            chars.entry(ch).or_insert(id);
        }
        if let Some(hex) = token.strip_prefix("<0x").and_then(|rest| rest.strip_suffix('>'))
            && hex.len() == 2
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            bytes[byte as usize] = id;
        }
    }
    if bytes.contains(&u32::MAX) {
        return Err("byte_fallback vocab is missing a <0xHH> token".to_string());
    }
    let mut rows = Vec::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let (left, right) = merge_sides(merge)?;
        let left_id = *vocab
            .get(left)
            .ok_or_else(|| format!("merge left {left:?} is not in the vocab"))?;
        let right_id = *vocab
            .get(right)
            .ok_or_else(|| format!("merge right {right:?} is not in the vocab"))?;
        let mut joined = String::with_capacity(left.len() + right.len());
        joined.push_str(left);
        joined.push_str(right);
        let id = *vocab
            .get(joined.as_str())
            .ok_or_else(|| format!("merge result {joined:?} is not in the vocab"))?;
        let rank = u32::try_from(rank).map_err(|_| "too many merges".to_string())?;
        rows.push((left_id, right_id, rank, id));
    }
    Ok(Kind::Spm {
        chars,
        bytes,
        pairs: Pairs::from_list(&rows),
    })
}

/// GPT-2 bytes-to-unicode. Printable ASCII and the latin-1 run stay themselves.
/// Every other byte is assigned the next scalar from U+0100.
fn byte_chars() -> &'static [char; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(bytes_to_unicode)
}

fn bytes_to_unicode() -> [char; 256] {
    let mut seen = [false; 256];
    let mut bs = Vec::with_capacity(256);
    for byte in b'!'..=b'~' {
        bs.push(byte);
    }
    for byte in 0xA1..=0xAC {
        bs.push(byte);
    }
    for byte in 0xAE..=0xFF {
        bs.push(byte);
    }
    for &byte in &bs {
        seen[byte as usize] = true;
    }
    let mut cs: Vec<u32> = bs.iter().map(|&byte| u32::from(byte)).collect();
    let mut next = 0u32;
    for byte in 0..256 {
        if !seen[byte] {
            bs.push(byte as u8);
            cs.push(256 + next);
            next += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (byte, scalar) in bs.into_iter().zip(cs) {
        out[byte as usize] = char::from_u32(scalar).expect("byte-level scalar");
    }
    out
}

fn wordpiece_of(
    model: &serde_json::Map<String, Value>,
    vocab_obj: &serde_json::Map<String, Value>,
) -> Result<Kind, String> {
    let mut rows = Vec::with_capacity(vocab_obj.len());
    for (token, id) in vocab_obj {
        rows.push((token.clone(), id_of(id)?));
    }
    let unk_token = model
        .get("unk_token")
        .and_then(|v| v.as_str())
        .ok_or("wordpiece has no unk_token")?;
    let unk = rows
        .iter()
        .find(|(token, _)| token == unk_token)
        .map(|(_, id)| *id)
        .ok_or_else(|| format!("unk {unk_token} is not in the vocab"))?;
    let prefix = model
        .get("continuing_subword_prefix")
        .and_then(|v| v.as_str())
        .unwrap_or("##")
        .to_string();
    let max_chars = model
        .get("max_input_chars_per_word")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;
    Ok(Kind::WordPiece {
        trie: VocabTrie::from_rows(rows),
        unk,
        prefix,
        max_chars,
    })
}

fn id_of(value: &Value) -> Result<u32, String> {
    let id = value.as_u64().ok_or_else(|| format!("vocab id is not an integer: {value}"))?;
    u32::try_from(id).map_err(|_| format!("vocab id {id} does not fit u32"))
}

fn merge_sides(value: &Value) -> Result<(&str, &str), String> {
    match value {
        Value::String(text) => text
            .split_once(' ')
            .ok_or_else(|| format!("merge {text:?} has no space")),
        Value::Array(sides) if sides.len() == 2 => {
            let left = sides[0].as_str().ok_or("merge left is not a string")?;
            let right = sides[1].as_str().ok_or("merge right is not a string")?;
            Ok((left, right))
        }
        _ => Err(format!("merge is not a pair: {value}")),
    }
}

/// Whitespace pieces with `▁` in front of each one, separated by a space
/// the T5 cut drops.
fn mark_words(text: &str, out: &mut String) {
    out.reserve(text.len() + text.len() / 8 + 3);
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut gap = false;
    let mut in_word = false;
    while i < bytes.len() {
        let b = bytes[i];
        // English is ASCII. The Unicode whitespace test and the char
        // decoder were most of the metaspace pass on that text.
        if b < 0x80 {
            if b == b' ' || b.wrapping_sub(9) < 5 {
                if in_word {
                    gap = true;
                    in_word = false;
                }
            } else {
                if !in_word {
                    if gap {
                        out.push(' ');
                    }
                    out.push('▁');
                    in_word = true;
                }
                out.push(b as char);
            }
            i += 1;
            continue;
        }
        let ch = text[i..].chars().next().expect("i is a char boundary");
        i += ch.len_utf8();
        if ch.is_whitespace() {
            if in_word {
                gap = true;
                in_word = false;
            }
            continue;
        }
        if !in_word {
            if gap {
                out.push(' ');
            }
            out.push('▁');
            in_word = true;
        }
        out.push(ch);
    }
}

fn decode_base64(text: &str) -> Result<Vec<u8>, String> {
    fn val(byte: u8) -> Result<u8, String> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err("precompiled charsmap is not base64".to_string()),
        }
    }
    let bytes: Vec<u8> = text.bytes().filter(|byte| !byte.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return Err("precompiled charsmap is not base64".to_string());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let pad = chunk.iter().rev().take_while(|byte| **byte == b'=').count();
        if pad > 2 {
            return Err("precompiled charsmap is not base64".to_string());
        }
        let mut n = [0u8; 4];
        for (i, &byte) in chunk.iter().enumerate() {
            n[i] = if byte == b'=' { 0 } else { val(byte)? };
        }
        out.push((n[0] << 2) | (n[1] >> 4));
        if pad < 2 {
            out.push((n[1] << 4) | (n[2] >> 2));
        }
        if pad < 1 {
            out.push((n[2] << 6) | n[3]);
        }
    }
    Ok(out)
}

fn default_threads() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn chunk_ranges(len: usize, n: usize) -> Vec<(usize, usize)> {
    let n = n.max(1).min(len);
    let base = len / n;
    let extra = len % n;
    let mut ranges = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..n {
        let size = base + usize::from(i < extra);
        ranges.push((start, start + size));
        start += size;
    }
    ranges
}

struct Fill {
    /// Ids written at the front of this worker's region.
    written: usize,
    /// `ends[i]` is the id count after document `i`, relative to `written`.
    ends: Vec<usize>,
    /// Set when `written` did not fit in the region. Holds every id for the chunk.
    spill: Option<Vec<u32>>,
    error: Option<String>,
}

fn fill_region(tokenizer: &mut Tokenizer, texts: &[&str], region: &mut [std::mem::MaybeUninit<u32>]) -> Fill {
    let mut scratch = Vec::new();
    let mut ends = Vec::with_capacity(texts.len());
    let mut at = 0usize;
    for text in texts {
        scratch.clear();
        if let Err(err) = tokenizer.encode(text, &mut scratch) {
            return Fill { written: at, ends, spill: None, error: Some(err.to_string()) };
        }
        if at + scratch.len() > region.len() {
            let mut spill = Vec::with_capacity(at + scratch.len());
            // SAFETY: `region[..at]` was initialized by earlier copies.
            spill.extend_from_slice(unsafe { std::slice::from_raw_parts(region.as_ptr().cast::<u32>(), at) });
            spill.extend_from_slice(&scratch);
            ends.push(spill.len());
            let done = ends.len();
            for text in texts.iter().skip(done) {
                if let Err(err) = tokenizer.encode(text, &mut spill) {
                    return Fill { written: 0, ends, spill: Some(spill), error: Some(err.to_string()) };
                }
                ends.push(spill.len());
            }
            return Fill { written: 0, ends, spill: Some(spill), error: None };
        }
        if !scratch.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    region.as_mut_ptr().add(at).cast::<u32>(),
                    scratch.len(),
                );
            }
        }
        at += scratch.len();
        ends.push(at);
    }
    Fill { written: at, ends, spill: None, error: None }
}

/// Slide each worker's ids to the front of `ids` and rebase document ends.
/// Stops at the first error, after including that chunk's finished documents.
fn place(ids: &mut Vec<u32>, ends: &mut Vec<usize>, base: usize, budgets: &[usize], fills: &[Fill]) -> Result<(), Error> {
    let keep = fills.iter().position(|fill| fill.error.is_some()).map(|i| i + 1).unwrap_or(fills.len());
    let any_spill = fills[..keep].iter().any(|fill| fill.spill.is_some());
    if any_spill {
        return place_spilled(ids, ends, base, budgets, &fills[..keep]);
    }
    let ptr = ids.as_mut_ptr();
    let mut cursor = base;
    let mut read = base;
    for (fill, budget) in fills[..keep].iter().zip(budgets) {
        if read != cursor && fill.written > 0 {
            // SAFETY: both ranges sit in the reserved tail and contain `written`
            // initialized ids. They can overlap, so this is a move.
            unsafe { std::ptr::copy(ptr.add(read), ptr.add(cursor), fill.written) };
        }
        for &end in &fill.ends {
            ends.push(cursor + end);
        }
        cursor += fill.written;
        read += budget;
    }
    // SAFETY: `base..cursor` is initialized. The tail past `cursor` stays spare.
    unsafe { ids.set_len(cursor) };
    if let Some(err) = fills[..keep].iter().rev().find_map(|fill| fill.error.clone()) {
        return Err(Error::new(err));
    }
    Ok(())
}

fn place_spilled(ids: &mut Vec<u32>, ends: &mut Vec<usize>, base: usize, budgets: &[usize], fills: &[Fill]) -> Result<(), Error> {
    let mut read = base;
    for (fill, budget) in fills.iter().zip(budgets) {
        let cursor = ids.len();
        if let Some(spill) = &fill.spill {
            ids.extend_from_slice(spill);
        } else if fill.written > 0 {
            let mut tmp = vec![0u32; fill.written];
            // SAFETY: `read..read+written` was initialized in the reserved tail.
            unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr().add(read), tmp.as_mut_ptr(), fill.written) };
            ids.extend_from_slice(&tmp);
        }
        for &end in &fill.ends {
            ends.push(cursor + end);
        }
        read += budget;
        if let Some(err) = &fill.error {
            return Err(Error::new(err.clone()));
        }
    }
    Ok(())
}

enum Form {
    Bytes,
    Pieces { prefix: String, cleanup: bool },
    Metaspace,
    Spm { strip_space: bool },
}

fn vocab_of(
    model: &serde_json::Map<String, Value>,
    added: Option<&Value>,
    decoder: Option<&Value>,
    kind: &Kind,
) -> Result<Vocab, String> {
    let form = form_of(decoder, kind);
    let entries = vocab_entries(model)?;
    let added = added_entries(added)?;
    let mut max = 0u32;
    for (_, id) in &entries {
        max = max.max(*id);
    }
    for (_, id, _) in &added {
        max = max.max(*id);
    }
    let n = max as usize + 1;
    let missing = Slice { at: u32::MAX, len: 0 };
    let mut vocab = Vocab {
        arena: Vec::new(),
        lead: vec![missing; n],
        next: vec![missing; n],
        tokens: vec![None; n],
        ids: FxHashMap::with_capacity_and_hasher(n, Default::default()),
        special: vec![false; n],
        strip_space: matches!(form, Form::Spm { strip_space: true }),
    };
    let map = unicode_to_byte();
    for (token, id) in entries {
        vocab.place(id, &token, &form, &map, false);
    }
    for (token, id, special) in added {
        vocab.place(id, &token, &form, &map, special);
    }
    Ok(vocab)
}

impl Vocab {
    fn place(&mut self, id: u32, token: &str, form: &Form, map: &FxHashMap<char, u8>, special: bool) {
        let index = id as usize;
        let (lead_bytes, next_bytes) = piece_forms(token, form, map);
        let lead = self.store(&lead_bytes);
        self.lead[index] = lead;
        self.next[index] = if next_bytes == lead_bytes {
            lead
        } else {
            self.store(&next_bytes)
        };
        self.ids.insert(token.to_string(), id);
        self.tokens[index] = Some(token.to_string());
        if special {
            self.special[index] = true;
        }
    }

    fn store(&mut self, bytes: &[u8]) -> Slice {
        let at = u32::try_from(self.arena.len()).expect("decode arena is longer than 4 GiB");
        let len = u32::try_from(bytes.len()).expect("one token is longer than 4 GiB");
        self.arena.extend_from_slice(bytes);
        Slice { at, len }
    }
}

fn piece_forms(token: &str, form: &Form, map: &FxHashMap<char, u8>) -> (Vec<u8>, Vec<u8>) {
    match form {
        Form::Bytes => {
            let bytes = byte_level_bytes(token, map);
            (bytes.clone(), bytes)
        }
        Form::Pieces { prefix, cleanup } => {
            let lead = if *cleanup { wordpiece_cleanup(token) } else { token.to_string() };
            let mut next = if let Some(rest) = token.strip_prefix(prefix.as_str()) {
                rest.to_string()
            } else {
                format!(" {token}")
            };
            if *cleanup {
                next = wordpiece_cleanup(&next);
            }
            (lead.into_bytes(), next.into_bytes())
        }
        Form::Metaspace => {
            let lead: String = token.chars().filter(|ch| *ch != '▁').collect();
            let next = token.replace('▁', " ");
            (lead.into_bytes(), next.into_bytes())
        }
        Form::Spm { .. } => {
            let bytes = spm_bytes(token);
            (bytes.clone(), bytes)
        }
    }
}

fn byte_level_bytes(token: &str, map: &FxHashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(token.len());
    for ch in token.chars() {
        let Some(&byte) = map.get(&ch) else {
            return token.as_bytes().to_vec();
        };
        out.push(byte);
    }
    out
}

fn spm_bytes(token: &str) -> Vec<u8> {
    if token.len() == 6
        && let Some(hex) = token.strip_prefix("<0x").and_then(|rest| rest.strip_suffix('>'))
        && let Ok(byte) = u8::from_str_radix(hex, 16)
    {
        return vec![byte];
    }
    token.replace('▁', " ").into_bytes()
}

fn wordpiece_cleanup(token: &str) -> String {
    token
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

pub(crate) fn unicode_to_byte() -> FxHashMap<char, u8> {
    let mut map = FxHashMap::default();
    for (byte, ch) in bytes_to_unicode().into_iter().enumerate() {
        map.insert(ch, byte as u8);
    }
    map
}

fn form_of(decoder: Option<&Value>, kind: &Kind) -> Form {
    if let Some(form) = decoder.and_then(read_form) {
        return form;
    }
    match kind {
        Kind::Bpe { .. } => Form::Bytes,
        Kind::WordPiece { prefix, .. } => Form::Pieces {
            prefix: prefix.clone(),
            cleanup: true,
        },
        Kind::Unigram { .. } => Form::Metaspace,
        Kind::Spm { .. } => Form::Spm { strip_space: true },
    }
}

fn read_form(value: &Value) -> Option<Form> {
    let obj = value.as_object()?;
    match obj.get("type").and_then(|v| v.as_str())? {
        "ByteLevel" => Some(Form::Bytes),
        "WordPiece" => Some(Form::Pieces {
            prefix: obj.get("prefix").and_then(|v| v.as_str()).unwrap_or("##").to_string(),
            cleanup: obj.get("cleanup").and_then(|v| v.as_bool()).unwrap_or(true),
        }),
        "Metaspace" => Some(Form::Metaspace),
        "Sequence" => sequence_form(obj.get("decoders")?.as_array()?),
        _ => None,
    }
}

fn sequence_form(list: &[Value]) -> Option<Form> {
    for item in list {
        if item.get("type").and_then(|v| v.as_str()) == Some("ByteLevel") {
            return Some(Form::Bytes);
        }
    }
    for item in list {
        if item.get("type").and_then(|v| v.as_str()) == Some("WordPiece") {
            return read_form(item);
        }
    }
    for item in list {
        if item.get("type").and_then(|v| v.as_str()) == Some("Metaspace") {
            return Some(Form::Metaspace);
        }
    }
    let marks = list.iter().any(replaces_metaspace);
    let bytes = list.iter().any(|item| item.get("type").and_then(|v| v.as_str()) == Some("ByteFallback"));
    let strip = list.iter().any(|item| item.get("type").and_then(|v| v.as_str()) == Some("Strip"));
    if marks || bytes {
        Some(Form::Spm { strip_space: strip })
    } else {
        None
    }
}

fn replaces_metaspace(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    if obj.get("type").and_then(|v| v.as_str()) != Some("Replace") {
        return false;
    }
    let pattern = obj.get("pattern").map(pattern_text).unwrap_or_default();
    pattern.contains('▁')
}

fn pattern_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(map) => map
            .get("String")
            .or_else(|| map.get("Regex"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

fn vocab_entries(model: &serde_json::Map<String, Value>) -> Result<Vec<(String, u32)>, String> {
    if let Some(list) = model.get("vocab").and_then(|v| v.as_array()) {
        let mut out = Vec::with_capacity(list.len());
        for (index, item) in list.iter().enumerate() {
            let token = item
                .as_array()
                .and_then(|pair| pair.first())
                .and_then(|v| v.as_str())
                .ok_or("unigram piece is not a string")?;
            out.push((token.to_string(), u32::try_from(index).map_err(|_| "unigram vocab is too large".to_string())?));
        }
        return Ok(out);
    }
    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or("model has no vocab")?;
    let mut out = Vec::with_capacity(vocab.len());
    for (token, id) in vocab {
        out.push((token.clone(), id_of(id)?));
    }
    Ok(out)
}

fn added_entries(value: Option<&Value>) -> Result<Vec<(String, u32, bool)>, String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let list = value.as_array().ok_or("added_tokens is not a list")?;
    let mut out = Vec::new();
    for token in list {
        let obj = token.as_object().ok_or("added token is not an object")?;
        let content = obj
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("added token has no content")?;
        if content.is_empty() {
            continue;
        }
        let id = obj
            .get("id")
            .and_then(|v| v.as_u64())
            .ok_or("added token has no id")?;
        let id = u32::try_from(id).map_err(|_| format!("added token id {id} does not fit u32"))?;
        let special = obj.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
        out.push((content.to_string(), id, special));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    const WITH: &str = r#"{
        "model": {
            "type": "BPE",
            "ignore_merges": true,
            "vocab": {".": 0, ":": 1, ".:": 2, ".:.:": 3},
            "merges": [". :"]
        }
    }"#;

    const WITHOUT: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {".": 0, ":": 1, ".:": 2, ".:.:": 3},
            "merges": [". :"]
        }
    }"#;

    #[test]
    fn ignore_merges_keeps_a_vocab_piece_whole() {
        let mut tok = Tokenizer::json(WITH).unwrap();
        let mut ids = Vec::new();
        tok.encode(".:.:", &mut ids).unwrap();
        assert_eq!(ids, [3]);
        ids.clear();
        tok.encode(".:.", &mut ids).unwrap();
        assert_eq!(ids, [2, 0]);

        let mut plain = Tokenizer::json(WITHOUT).unwrap();
        ids.clear();
        plain.encode(".:.:", &mut ids).unwrap();
        assert_eq!(ids, [2, 2]);

        let path = std::env::temp_dir().join(format!("tokie-ignore-{}.tokie", std::process::id()));
        tok.save(&path).unwrap();
        let mut loaded = Tokenizer::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        ids.clear();
        loaded.encode(".:.:", &mut ids).unwrap();
        assert_eq!(ids, [3]);
    }

    fn sequential(tok: &mut Tokenizer, docs: &[&str]) -> (Vec<u32>, Vec<usize>) {
        let mut ids = Vec::new();
        let mut ends = Vec::new();
        for doc in docs {
            tok.encode(doc, &mut ids).unwrap();
            ends.push(ids.len());
        }
        (ids, ends)
    }

    #[test]
    fn batch_matches_sequential_encode() {
        let mut tok = Tokenizer::json(WITH).unwrap();
        let long = ".".repeat(300);
        let owned: Vec<String> = (0..40).map(|i| ".".repeat(i % 7 + 1)).collect();
        let mut docs: Vec<&str> = owned.iter().map(String::as_str).collect();
        docs.push(long.as_str());
        docs.push(".:.:");
        docs.push("");

        let (expect, expect_ends) = sequential(&mut tok, &docs);

        tok.threads(4);
        let mut ids = vec![99];
        let mut ends = Vec::new();
        tok.batch(&docs, &mut ids, &mut ends).unwrap();
        assert_eq!(ids[0], 99);
        assert_eq!(&ids[1..], expect.as_slice());
        assert_eq!(ends.len(), docs.len());
        for (end, expect_end) in ends.iter().zip(&expect_ends) {
            assert_eq!(*end, expect_end + 1);
        }

        ids.clear();
        ends.clear();
        tok.batch(&docs, &mut ids, &mut ends).unwrap();
        assert_eq!(ids, expect);
        assert_eq!(ends, expect_ends);

        tok.threads(1);
        ids.clear();
        ends.clear();
        tok.batch(&docs, &mut ids, &mut ends).unwrap();
        assert_eq!(ids, expect);
        assert_eq!(ends, expect_ends);

        let n = ids.len();
        tok.batch(&[], &mut ids, &mut ends).unwrap();
        assert_eq!(ids.len(), n);
    }
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Tokenizer>();
};
