//! Generic pretokenizer iterator parameterized by `PretokConfig`.
//!
//! All BPE-style pretokenizers (GPT-2, CL100K, O200K, Voyage, SmolLM, DeepSeek, Qwen)
//! are monomorphizations of `Core<C>` with different config constants. The compiler
//! const-propagates all config checks and dead-code-eliminates unreachable branches,
//! producing identical machine code to hand-written per-type implementations.

use std::marker::PhantomData;

use crate::core::config::*;
use crate::util::{is_cjk_char, ascii_letter_run, decode_utf8, is_ascii_letter, is_digit, is_lower, is_punct_or_symbol, is_tab_like, is_upper, is_unicode_letter, is_unicode_mark};

pub struct Core<'a, C: PretokConfig> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
    _cfg: PhantomData<C>,
}

impl<'a, C: PretokConfig> Core<'a, C> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len(), _cfg: PhantomData }
    }

    /// Resume iteration at byte offset `pos`, which must be a piece boundary.
    /// Used by the mask scanner to re-derive pieces inside bad zones.
    pub fn with_pos(text: &'a str, pos: usize) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos, len: bytes.len(), _cfg: PhantomData }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }

    // ---- Letter scanning ----

    /// Scan letters: `\p{L}+` (Plain) or `[\p{L}\p{M}]+` (PlainWithMarks).
    /// Not used for CamelCase — that calls `scan_letters_case_aware` instead.
    #[inline(always)]
    fn scan_letters(&mut self) {
        loop {
            // SWAR fast path: consume whole ASCII-letter runs 8 bytes at a time
            self.pos += ascii_letter_run(&self.bytes[self.pos..self.len]);
            if self.pos >= self.len {
                return;
            }
            let b = self.at(self.pos);
            if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if Self::isolated_cjk(ch) {
                    return;
                } else if C::LETTER_MODE == LetterMode::PlainWithMarks {
                    if crate::util::is_alpha(ch) || is_unicode_mark(ch) {
                        self.pos += cl;
                    } else { return; }
                } else {
                    if is_unicode_letter(ch) { self.pos += cl; } else { return; }
                }
            } else {
                return;
            }
        }
    }

    /// O200K CamelCase, entered with the first letter byte (ASCII) consumed.
    #[inline(always)]
    fn scan_letters_case_aware(&mut self, _first: u8) {
        self.pos -= 1;
        self.scan_camel();
    }

    /// Case class of the letter/mark char at `pos` for the CamelCase rule:
    /// bit 0 = in `U = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, bit 1 = in
    /// `L = [\p{Ll}\p{Lm}\p{Lo}\p{M}]` (Lm/Lo/M are in both), 0 = neither.
    /// Returns (class, char byte length).
    #[inline(always)]
    fn camel_class_at(&self, pos: usize) -> (u8, usize) {
        let b = self.at(pos);
        if is_upper(b) {
            (1, 1)
        } else if is_lower(b) {
            (2, 1)
        } else if b < 0x80 {
            (0, 1)
        } else {
            let (ch, cl) = decode_utf8(&self.bytes[pos..]);
            (crate::util::camel_class(ch), cl)
        }
    }

    /// `U*L+|U+L*` (the o200k/Tekken word body) from the letter char at
    /// `pos`, with the regex's backtracking: the greedy `U*` run gives back
    /// chars until `L+` can start, so a run with no following `Ll` ends
    /// after its last `L`-class char (`한Q` → `한`, `Q`), and an all-`Lu`/`Lt`
    /// run is taken whole by `U+`.
    #[inline(always)]
    fn scan_camel(&mut self) {
        let (c0, cl0) = self.camel_class_at(self.pos);
        self.scan_camel_known(c0, cl0);
    }

    /// [`Self::scan_camel`] from a letter/mark char `ch` already decoded at
    /// `pos` by the caller's letter check (no second `is_alpha`).
    #[inline(always)]
    fn scan_camel_letter(&mut self, ch: char) {
        self.scan_camel_known(crate::util::camel_class_of_letter(ch), ch.len_utf8());
    }

    /// [`Self::scan_camel`] given the class `c0` and byte length `cl0` of
    /// the char at `pos`.
    #[inline(always)]
    fn scan_camel_known(&mut self, c0: u8, cl0: usize) {
        let mut p = self.pos;
        let mut last_l_end = 0usize;
        // `U*`: ASCII uppercase in a tight loop (never in L), unicode by
        // class. Ends with the class/length of the char that stopped it, so
        // no char is classified twice.
        let (next_class, next_len) = if c0 & 1 == 0 {
            (c0, cl0)
        } else {
            p += cl0;
            if c0 & 2 != 0 {
                last_l_end = p;
            }
            loop {
                while p < self.len && is_upper(self.at(p)) {
                    p += 1;
                }
                if p >= self.len {
                    break (0, 0);
                }
                let b = self.at(p);
                if b < 0x80 {
                    break (if is_lower(b) { 2 } else { 0 }, 1);
                }
                let (c, cl) = self.camel_class_at(p);
                if c & 1 == 0 {
                    break (c, cl);
                }
                p += cl;
                if c & 2 != 0 {
                    last_l_end = p;
                }
            }
        };
        if next_class == 2 {
            // `L+` continues the run past the uppercase prefix.
            p += next_len;
            loop {
                while p < self.len && is_lower(self.at(p)) {
                    p += 1;
                }
                if p >= self.len || self.at(p) < 0x80 {
                    break;
                }
                let (c, cl) = self.camel_class_at(p);
                if c & 2 == 0 {
                    break;
                }
                p += cl;
            }
            self.pos = p;
        } else if last_l_end != 0 {
            self.pos = last_l_end;
        } else {
            self.pos = p;
        }
    }

    /// Scan ASCII-only letters (DeepSeek: punct prefix → ASCII letters only).
    #[inline(always)]
    fn scan_ascii_letters(&mut self) {
        self.pos += ascii_letter_run(&self.bytes[self.pos..self.len]);
    }

    // ---- Digit scanning ----

    /// Scan remaining digits after the first digit byte is already consumed.
    #[inline(always)]
    fn scan_digits(&mut self) {
        // \p{N} covers Nd/Nl/No, so non-ASCII numerics (¹, ❶, Ⅷ) continue a run
        match C::DIGIT_MODE {
            DigitMode::Unlimited => {
                while self.pos < self.len {
                    let b = self.at(self.pos);
                    if is_digit(b) {
                        self.pos += 1;
                    } else if b >= 0x80 {
                        let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_numeric() { self.pos += cl; } else { break; }
                    } else {
                        break;
                    }
                }
            }
            DigitMode::Chunked3 => {
                // First digit already consumed; scan up to 2 more chars
                let mut count = 1u8;
                while self.pos < self.len && count < 3 {
                    let b = self.at(self.pos);
                    if is_digit(b) {
                        self.pos += 1;
                        count += 1;
                    } else if b >= 0x80 {
                        let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_numeric() { self.pos += cl; count += 1; } else { break; }
                    } else {
                        break;
                    }
                }
            }
            DigitMode::Single => {
                // Single digit already consumed by caller
            }
        }
    }

    // ---- Contraction detection ----

    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if C::CONTRACTION_MODE == ContractionMode::None { return 0; }
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = if C::CONTRACTION_CASE == ContractionCase::Insensitive {
            self.bytes[self.pos + 1] | 0x20
        } else {
            self.bytes[self.pos + 1]
        };
        // No lookahead in the regex: '(?i:[sdmt]|ll|ve|re) matches even when
        // more letters follow ("don'ts" -> "don", "'t", "s"; "O'Toole" -> "O", "'T", "oole")
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            return 2;
        }
        if rem < 3 { return 0; }
        let b2 = if C::CONTRACTION_CASE == ContractionCase::Insensitive {
            self.bytes[self.pos + 2] | 0x20
        } else {
            self.bytes[self.pos + 2]
        };
        if (b1 == b'l' && b2 == b'l')
            || (b1 == b'v' && b2 == b'e')
            || (b1 == b'r' && b2 == b'e')
        {
            return 3;
        }
        0
    }

    // ---- Punctuation scanning ----

    #[inline(always)]
    fn is_punct_byte(b: u8) -> bool {
        if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly {
            // [\p{P}\p{S}] in ASCII = printable non-alphanumerics; excludes controls
            matches!(b, 0x21..=0x2F | 0x3A..=0x40 | 0x5B..=0x60 | 0x7B..=0x7E)
        } else {
            !is_ascii_letter(b) && !is_digit(b) && b != b' ' && !is_tab_like(b) && b != b'\n' && b != b'\r' && b < 0x80
        }
    }

    /// Check if a Unicode char is in this config's punct class.
    #[inline(always)]
    fn is_unicode_punct_char(ch: char) -> bool {
        if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly {
            is_punct_or_symbol(ch) && !Self::isolated_cjk(ch)
        } else if C::LETTER_MODE == LetterMode::PlainWithMarks {
            // Qwen3.5 ` ?[^\s\p{L}\p{M}\p{N}]+`. O200K's ` ?[^\s\p{L}\p{N}]+`
            // keeps marks in the run (" ❤️" is one piece), so it takes the
            // default branch.
            !crate::util::is_alpha(ch) && !ch.is_numeric() && !ch.is_whitespace() && !is_unicode_mark(ch)
        } else {
            !is_unicode_letter(ch) && !ch.is_numeric() && !ch.is_whitespace()
        }
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        // Scan punct chars
        while self.pos < self.len {
            let b = self.at(self.pos);
            if Self::is_punct_byte(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if Self::is_unicode_punct_char(ch) {
                    self.pos += cl;
                } else { break; }
            } else { break; }
        }
        // Trailing newlines
        if C::PUNCT_TRAILING == PunctTrailing::Newlines || C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes {
            while self.pos < self.len {
                let b = self.at(self.pos);
                if b == b'\n' || b == b'\r' {
                    self.pos += 1;
                } else if C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes && b == b'/' {
                    self.pos += 1;
                } else { break; }
            }
        }
    }

    // ---- Whitespace scanning ----

    #[inline(always)]
    fn scan_whitespace(&mut self) {
        if C::WS_PATTERN == WsPattern::Cl100k {
            self.scan_whitespace_cl100k();
        } else {
            self.scan_whitespace_gpt2();
        }
    }

    /// GPT-2 style: `\s+(?!\S)|\s+` — greedy, back up one if followed by non-WS.
    #[inline(always)]
    fn scan_whitespace_gpt2(&mut self) {
        let start = self.pos;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b' ' || c == b'\n' || c == b'\r' || is_tab_like(c) {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        if self.pos < self.len && prev_pos > start {
            if C::WS_EXCEPTION == WsException::Digits {
                // SmolLM: don't back up before numerics (\p{N} incl. ½, ¹)
                let next = self.at(self.pos);
                let is_num = is_digit(next)
                    || (next >= 0x80 && decode_utf8(&self.bytes[self.pos..]).0.is_numeric());
                if !is_num {
                    self.pos = prev_pos;
                }
            } else {
                self.pos = prev_pos;
            }
        }
    }

    /// CL100K style: `\s*[\r\n]|\s+(?!\S)|\s+` — prioritize newlines.
    #[inline(always)]
    fn scan_whitespace_cl100k(&mut self) {
        let start = self.pos;
        let mut last_newline_end = 0usize;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b'\n' || c == b'\r' {
                prev_pos = self.pos;
                self.pos += 1;
                last_newline_end = self.pos;
            } else if c == b' ' || is_tab_like(c) {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        if last_newline_end > 0 {
            self.pos = last_newline_end;
        } else if self.pos < self.len && prev_pos > start {
            if C::WS_EXCEPTION == WsException::Cjk {
                // DeepSeek: don't back up before CJK or any \p{N} (incl. ½, ¹):
                // both are split off by earlier Split stages, so the run ends
                // its segment
                let next = self.at(self.pos);
                let split_off = is_digit(next)
                    || (next >= 0x80
                        && (Self::is_cjk_start(&self.bytes[self.pos..])
                            || decode_utf8(&self.bytes[self.pos..]).0.is_numeric()));
                if !split_off {
                    self.pos = prev_pos;
                }
            } else {
                self.pos = prev_pos;
            }
        }
    }

    /// Check if bytes start a CJK character.
    #[inline(always)]
    fn is_cjk_start(bytes: &[u8]) -> bool {
        if bytes[0] < 0xE0 { return false; }
        is_cjk_char(decode_utf8(bytes).0)
    }

    /// DeepSeek pre-splits `[一-龥぀-ゟ゠-ヿ]+` (Isolated) before its main
    /// regex, so no piece crosses the edge of such a run (`・-R` → `・`,
    /// `-R`; `Ｆ一` → `Ｆ`, `一`; ` 中` → ` `, `中`).
    #[inline(always)]
    fn isolated_cjk(ch: char) -> bool {
        C::WS_EXCEPTION == WsException::Cjk && is_cjk_char(ch)
    }

    /// Inside a CJK run the main regex still applies: a piece is a run of
    /// letters/marks or of punct/symbols (`ア゛` → `ア`, `゛`), with no
    /// prefixes (the run holds no ASCII or whitespace). `first` is consumed.
    #[inline(always)]
    fn scan_cjk(&mut self, first: char) {
        let letter = |c: char| crate::util::is_alpha(c) || is_unicode_mark(c);
        let want = letter(first);
        while self.pos < self.len && self.at(self.pos) >= 0xE0 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            if !is_cjk_char(ch) || letter(ch) != want { break; }
            self.pos += cl;
        }
    }

    // ---- Letter entry helpers (handles contraction after letters) ----

    /// After scanning letters, handle contraction check based on config.
    /// Returns true if we should early-return (standalone contraction found).
    #[inline(always)]
    fn handle_post_letters(&mut self, start: usize) -> Option<&'a str> {
        if C::CONTRACTION_MODE == ContractionMode::Suffix {
            let clen = self.check_contraction();
            if clen > 0 { self.pos += clen; }
            None // suffix: contraction merged, no early return
        } else {
            // Standalone: if contraction follows, DON'T consume it — emit word only
            if self.check_contraction() > 0 {
                Some(self.emit(start))
            } else {
                None
            }
        }
    }

    /// Scan letters starting from current position (first letter byte already consumed).
    #[inline(always)]
    fn do_letter_scan(&mut self, first: u8) {
        if C::LETTER_MODE == LetterMode::CamelCase {
            self.scan_letters_case_aware(first);
        } else {
            self.scan_letters();
        }
    }

    /// Check if a Unicode char is a letter for this config.
    #[inline(always)]
    fn is_letter_char(ch: char) -> bool {
        if Self::isolated_cjk(ch) {
            false
        } else if C::LETTER_MODE == LetterMode::PlainWithMarks || C::LETTER_MODE == LetterMode::CamelCase {
            crate::util::is_alpha(ch) || is_unicode_mark(ch)
        } else {
            is_unicode_letter(ch)
        }
    }

    // ---- Punct prefix: scan letters after a prefix char ----

    /// Scan the letter run whose first char is at `self.pos`, right after a
    /// one-char prefix. Case-aware under CamelCase, where the o200k
    /// contraction suffix is merged too.
    #[inline(always)]
    fn scan_unicode_letters_after_prefix(&mut self, ch: char) {
        if C::LETTER_MODE == LetterMode::CamelCase {
            self.scan_camel_letter(ch);
            let clen = self.check_contraction();
            if clen > 0 { self.pos += clen; }
        } else {
            self.scan_letters();
        }
    }

    /// After a non-alnum prefix char, scan following letters.
    /// DeepSeek: only ASCII letters. Others: full unicode letters.
    #[inline(always)]
    fn scan_letters_after_punct_prefix(&mut self, next: u8) {
        if C::PUNCT_PREFIX_MODE == PunctPrefixMode::AsciiOnly {
            self.pos += 1; // consume the ASCII letter
            self.scan_ascii_letters();
        } else {
            self.pos += 1; // consume the letter byte
            self.do_letter_scan(next);
        }
    }
}

impl<'a, C: PretokConfig> Core<'a, C> {
    /// Whether the tab at `self.pos` attaches to a following letter run.
    #[inline(always)]
    fn tab_prefixes_letters(&self) -> bool {
        if C::PUNCT_PREFIX_MODE == PunctPrefixMode::SpaceOnly || self.pos + 1 >= self.len {
            return false;
        }
        let next = self.at(self.pos + 1);
        is_ascii_letter(next) || (next >= 0x80 && Self::is_letter_char(decode_utf8(&self.bytes[self.pos + 1..]).0))
    }

    /// Whether a letter char (this config's letter class) starts at `pos`.
    #[inline(always)]
    fn letter_at(&self, pos: usize) -> bool {
        pos < self.len && {
            let b = self.at(pos);
            is_ascii_letter(b) || (b >= 0x80 && Self::is_letter_char(decode_utf8(&self.bytes[pos..]).0))
        }
    }

    /// DeepSeek: byte length of the char at `pos` if no main-regex
    /// alternative can start on it (Cc/Cf/Co/Cn — not letter, mark, digit,
    /// whitespace, `[\p{P}\p{S}]` or a pre-split CJK char), else 0.
    #[inline(always)]
    fn gap_char_at(&self, pos: usize) -> usize {
        if pos >= self.len {
            return 0;
        }
        let b = self.at(pos);
        if b < 0x80 {
            return usize::from((b < 0x20 && !is_tab_like(b) && b != b'\n' && b != b'\r') || b == 0x7F);
        }
        let (ch, cl) = decode_utf8(&self.bytes[pos..]);
        let gap = !is_cjk_char(ch)
            && !crate::util::is_alpha(ch)
            && !is_unicode_mark(ch)
            && !ch.is_numeric()
            && !ch.is_whitespace()
            && !is_punct_or_symbol(ch);
        if gap { cl } else { 0 }
    }

    /// After a consumed non-punct control char (DeepSeek): letters take it
    /// as their one-char prefix (`[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`).
    /// Otherwise nothing matches it, and `Isolated` keeps the whole
    /// unmatched gap as one piece ("\x1b\x1b" stays together) — up to a
    /// gap char that prefixes the letters after it ("\x1b\x1bab" →
    /// "\x1b", "\x1bab").
    #[inline(always)]
    fn scan_after_control_prefix(&mut self, start: usize) -> Option<&'a str> {
        if self.letter_at(self.pos) {
            let next = self.at(self.pos);
            self.pos += if next < 0x80 { 1 } else { decode_utf8(&self.bytes[self.pos..]).1 };
            self.scan_letters();
            return self.handle_post_letters(start);
        }
        loop {
            let gl = self.gap_char_at(self.pos);
            if gl == 0 || self.letter_at(self.pos + gl) {
                return None;
            }
            self.pos += gl;
        }
    }
}

// ---- Iterator implementation ----

impl<'a, C: PretokConfig> Iterator for Core<'a, C> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            // Letter start
            self.pos += 1;
            self.do_letter_scan(b);
            if let Some(piece) = self.handle_post_letters(start) {
                return Some(piece);
            }
        } else if b == b'\'' {
            // Apostrophe: contraction or punct prefix
            let clen = self.check_contraction();
            if clen > 0 && C::CONTRACTION_MODE == ContractionMode::Standalone {
                self.pos += clen;
            } else if C::PUNCT_PREFIX_MODE == PunctPrefixMode::SpaceOnly {
                // GPT-2 ` ?[^\s\p{L}\p{N}]+`: a non-contraction apostrophe is
                // punct and never prefixes letters ("d'Arce" → d ' Arce).
                self.pos += 1;
                self.scan_punct();
            } else if clen == 0 || C::CONTRACTION_MODE == ContractionMode::Suffix {
                // Not a standalone contraction — treat as prefix or punct
                if self.pos + 1 < self.len {
                    let next = self.at(self.pos + 1);
                    if is_ascii_letter(next) {
                        self.pos += 1; // skip apostrophe prefix
                        self.scan_letters_after_punct_prefix(next);
                        if let Some(piece) = self.handle_post_letters(start) {
                            return Some(piece);
                        }
                    } else if next >= 0x80 {
                        let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                        if Self::is_letter_char(ch) && C::PUNCT_PREFIX_MODE != PunctPrefixMode::AsciiOnly {
                            self.pos += 1; // skip apostrophe prefix
                            self.scan_unicode_letters_after_prefix(ch);
                            if C::LETTER_MODE != LetterMode::CamelCase {
                                if let Some(piece) = self.handle_post_letters(start) {
                                    return Some(piece);
                                }
                            }
                        } else {
                            self.pos += 1;
                            self.scan_punct();
                        }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                }
            }
        } else if is_digit(b) {
            if C::DIGIT_MODE == DigitMode::Single {
                self.pos += 1;
            } else {
                self.pos += 1;
                self.scan_digits();
            }
        } else if is_tab_like(b) && !self.tab_prefixes_letters() {
            // Tab (and VT/FF) only ever prefixes letters (`[^\r\n\p{L}\p{N}]?\p{L}+`);
            // the punct/digit rules take a literal space (` ?`), and GPT-2's
            // ` ?\p{L}+` takes no tab at all. Otherwise it's plain whitespace.
            self.scan_whitespace();
        } else if b == b' ' || is_tab_like(b) {
            // Space/tab: prefix letters, prefix punct, or whitespace run
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 2;
                    self.do_letter_scan(next);
                    if let Some(piece) = self.handle_post_letters(start) {
                        return Some(piece);
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if Self::is_letter_char(ch) {
                        self.pos += 1; // consume space prefix
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            self.scan_camel_letter(ch);
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else {
                            self.scan_letters();
                            if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        }
                    } else if ch.is_numeric() && C::SPACE_PREFIXES_DIGITS {
                        // GPT-2 ` ?\p{N}+`: space attaches to a numeric run (incl. ¹, ❶)
                        let (_, cl) = decode_utf8(&self.bytes[self.pos + 1..]);
                        self.pos += 1 + cl;
                        self.scan_digits();
                    } else if ch.is_whitespace() || ch.is_numeric() || Self::isolated_cjk(ch) {
                        self.scan_whitespace();
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else if C::SPACE_PREFIXES_DIGITS && is_digit(next) {
                    // GPT-2: space prefixes digits
                    self.pos += 2;
                    self.scan_digits();
                } else if Self::is_punct_byte(next) || next == b'\'' {
                    if C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly || next == b'\'' {
                        // CL100K/O200K/Voyage/Qwen: space prefixes punct
                        self.pos += 1;
                        self.scan_punct();
                    } else {
                        // GPT-2/SmolLM: space + punct → space prefixes punct group
                        self.pos += 2;
                        self.scan_punct();
                    }
                } else if is_digit(next) {
                    // Space before digit, no prefix (CL100K/Voyage/SmolLM etc.)
                    if C::WS_PATTERN == WsPattern::Cl100k {
                        self.scan_whitespace();
                    } else {
                        // GPT-2 style: SmolLM emits bare space before digit
                        self.pos += 1;
                    }
                } else {
                    // Space + more whitespace
                    self.scan_whitespace();
                }
            } else {
                self.pos += 1;
            }
        } else if b == b'\n' || b == b'\r' {
            if C::WS_PATTERN == WsPattern::Cl100k {
                // CL100K `\s*[\r\n]+`: through the last newline of the whole
                // whitespace run ("\r\n  \r\n  x" → "\r\n  \r\n", " ", " x").
                self.scan_whitespace();
            } else {
                // GPT-2: whitespace run with lookahead
                self.scan_whitespace();
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            if Self::isolated_cjk(ch) {
                self.pos += cl;
                self.scan_cjk(ch);
            } else if Self::is_letter_char(ch) {
                if C::LETTER_MODE == LetterMode::CamelCase {
                    self.scan_camel_letter(ch);
                } else {
                    self.pos += cl;
                    self.scan_letters();
                }
                if let Some(piece) = self.handle_post_letters(start) {
                    return Some(piece);
                }
            } else if is_unicode_mark(ch) && C::LETTER_MODE == LetterMode::CamelCase {
                // O200K: mark at start (in both case classes)
                self.scan_camel_letter(ch);
                let clen = self.check_contraction();
                if clen > 0 { self.pos += clen; }
            } else if ch.is_numeric() {
                self.pos += cl;
                self.scan_digits();
            } else if ch.is_whitespace() {
                // `[^\r\n\p{L}\p{N}]?\p{L}+`: non-ASCII whitespace (NBSP, U+3000)
                // is a valid one-char letter prefix, like tab.
                // DeepSeek's `[^\r\n\p{L}\p{P}\p{S}]?` admits it as well;
                // GPT-2's ` ?\p{L}+` takes a literal space only.
                let next_letter = C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly
                    && self.pos + cl < self.len
                    && {
                        let next = self.at(self.pos + cl);
                        is_ascii_letter(next)
                            || (next >= 0x80 && Self::is_letter_char(decode_utf8(&self.bytes[self.pos + cl..]).0))
                    };
                if next_letter {
                    self.pos += cl;
                    let next = self.at(self.pos);
                    if is_ascii_letter(next) {
                        // Full letter run (not DeepSeek's ASCII-only punct rule).
                        self.pos += 1;
                        self.do_letter_scan(next);
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else if let Some(piece) = self.handle_post_letters(start) {
                            return Some(piece);
                        }
                    } else {
                        let (ch2, _) = decode_utf8(&self.bytes[self.pos..]);
                        self.scan_unicode_letters_after_prefix(ch2);
                        if C::LETTER_MODE != LetterMode::CamelCase {
                            if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        }
                    }
                } else {
                    self.scan_whitespace();
                }
            } else if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly && !is_punct_or_symbol(ch) {
                // DeepSeek: a Cf/Cc char is not punct — it's the letter rule's
                // optional one-char prefix [^\r\n\p{L}\p{P}\p{S}]?, or stands alone
                self.pos += cl;
                if let Some(piece) = self.scan_after_control_prefix(start) {
                    return Some(piece);
                }
            } else {
                // Non-ASCII symbol: one-char prefix to letters where the config's
                // letter rule allows it, else a punct group (incl. trailing newlines)
                self.pos += cl;
                if C::PUNCT_PREFIX_MODE == PunctPrefixMode::SpaceOnly
                    || C::PUNCT_PREFIX_MODE == PunctPrefixMode::AsciiOnly
                {
                    // GPT-2/SmolLM: punct never prefixes letters.
                    // DeepSeek: non-ASCII \p{P}\p{S} is excluded from the letter-prefix
                    // class — only the ASCII punct list prefixes letters.
                    self.scan_punct();
                } else if self.pos < self.len {
                    let next = self.at(self.pos);
                    if is_ascii_letter(next) {
                        self.scan_letters_after_punct_prefix(next);
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else if let Some(piece) = self.handle_post_letters(start) {
                            return Some(piece);
                        }
                    } else if next >= 0x80 {
                        let (ch2, _) = decode_utf8(&self.bytes[self.pos..]);
                        if Self::is_letter_char(ch2) {
                            self.scan_unicode_letters_after_prefix(ch2);
                        } else {
                            self.scan_punct();
                        }
                    } else {
                        self.scan_punct();
                    }
                }
            }
        } else if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly && !Self::is_punct_byte(b) {
            // DeepSeek: an ASCII control (ESC, NUL, DEL) is not punct either
            self.pos += 1;
            if let Some(piece) = self.scan_after_control_prefix(start) {
                return Some(piece);
            }
        } else {
            // Other ASCII punct
            if C::PUNCT_PREFIX_MODE == PunctPrefixMode::SpaceOnly {
                // GPT-2/SmolLM: punct doesn't prefix letters, just scan punct group
                self.pos += 1;
                self.scan_punct();
            } else if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 1; // skip punct prefix
                    self.scan_letters_after_punct_prefix(next);
                    if C::LETTER_MODE == LetterMode::CamelCase {
                        let clen = self.check_contraction();
                        if clen > 0 { self.pos += clen; }
                    } else if let Some(piece) = self.handle_post_letters(start) {
                        return Some(piece);
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    // DeepSeek: ASCII punct prefixes `[A-Za-z]+` only ("=σλ" → = σλ)
                    if Self::is_letter_char(ch) && C::PUNCT_PREFIX_MODE != PunctPrefixMode::AsciiOnly {
                        self.pos += 1; // skip punct prefix
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            self.scan_camel_letter(ch);
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else {
                            self.scan_letters();
                            if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                    self.scan_punct();
                }
            } else {
                self.pos += 1;
            }
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::*;

    type Gpt2<'a> = Core<'a, Gpt2Config>;
    type Cl100k<'a> = Core<'a, Cl100kConfig>;
    type O200k<'a> = Core<'a, O200kConfig>;
    type Voyage<'a> = Core<'a, VoyageConfig>;
    type SmolLM<'a> = Core<'a, SmolLMConfig>;
    type DeepSeek<'a> = Core<'a, DeepSeekConfig>;
    type Qwen<'a> = Core<'a, QwenConfig>;
    type Tekken<'a> = Core<'a, TekkenConfig>;

    // GPT-2
    #[test] fn gpt2_basic() { assert_eq!(Gpt2::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn gpt2_punct() { assert_eq!(Gpt2::new("Hello, world!").collect::<Vec<_>>(), vec!["Hello", ",", " world", "!"]); }
    #[test] fn gpt2_numbers() { assert_eq!(Gpt2::new("test 123").collect::<Vec<_>>(), vec!["test", " 123"]); }
    #[test] fn gpt2_contraction() { assert_eq!(Gpt2::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn gpt2_contraction_ll() { assert_eq!(Gpt2::new("I'll").collect::<Vec<_>>(), vec!["I", "'ll"]); }
    #[test] fn gpt2_contraction_ve() { assert_eq!(Gpt2::new("I've").collect::<Vec<_>>(), vec!["I", "'ve"]); }
    #[test] fn gpt2_contraction_re() { assert_eq!(Gpt2::new("we're").collect::<Vec<_>>(), vec!["we", "'re"]); }
    #[test] fn gpt2_space_digit() { assert_eq!(Gpt2::new("x 42").collect::<Vec<_>>(), vec!["x", " 42"]); }
    #[test] fn gpt2_double_newline() { assert_eq!(Gpt2::new("a\n\nb").collect::<Vec<_>>(), vec!["a", "\n", "\n", "b"]); }
    #[test] fn gpt2_newline_spaces() { assert_eq!(Gpt2::new("a\n  b").collect::<Vec<_>>(), vec!["a", "\n ", " b"]); }
    #[test] fn gpt2_multi_spaces() { assert_eq!(Gpt2::new("a  b").collect::<Vec<_>>(), vec!["a", " ", " b"]); }
    #[test] fn gpt2_space_punct() { assert_eq!(Gpt2::new("a <b").collect::<Vec<_>>(), vec!["a", " <", "b"]); }
    #[test] fn gpt2_triple_spaces() { assert_eq!(Gpt2::new("a   b").collect::<Vec<_>>(), vec!["a", "  ", " b"]); }

    // CL100K
    #[test] fn cl100k_basic() { assert_eq!(Cl100k::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn cl100k_digits() { assert_eq!(Cl100k::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn cl100k_case_insensitive() { assert_eq!(Cl100k::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]); }
    #[test] fn cl100k_punct_prefix() { assert_eq!(Cl100k::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }
    #[test] fn cl100k_space_prefix() { assert_eq!(Cl100k::new("a b").collect::<Vec<_>>(), vec!["a", " b"]); }
    #[test] fn cl100k_newline() { assert_eq!(Cl100k::new("a\nb").collect::<Vec<_>>(), vec!["a", "\n", "b"]); }
    #[test] fn cl100k_space_newline() { assert_eq!(Cl100k::new("a \nb").collect::<Vec<_>>(), vec!["a", " \n", "b"]); }

    // O200K
    #[test] fn o200k_basic() { assert_eq!(O200k::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn o200k_suffix() { assert_eq!(O200k::new("don't").collect::<Vec<_>>(), vec!["don't"]); }
    #[test] fn o200k_suffix_cases() {
        assert_eq!(O200k::new("DON'T").collect::<Vec<_>>(), vec!["DON'T"]);
        assert_eq!(O200k::new("I'm").collect::<Vec<_>>(), vec!["I'm"]);
        assert_eq!(O200k::new("we'll").collect::<Vec<_>>(), vec!["we'll"]);
    }
    #[test] fn o200k_camelcase() {
        assert_eq!(O200k::new("CamelCase").collect::<Vec<_>>(), vec!["Camel", "Case"]);
        assert_eq!(O200k::new("JSONParser").collect::<Vec<_>>(), vec!["JSONParser"]);
        assert_eq!(O200k::new("parseJSON").collect::<Vec<_>>(), vec!["parse", "JSON"]);
        assert_eq!(O200k::new("XMLHttpRequest").collect::<Vec<_>>(), vec!["XMLHttp", "Request"]);
    }
    #[test] fn o200k_digits() { assert_eq!(O200k::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn o200k_apost_prefix() { assert_eq!(O200k::new("'hello'").collect::<Vec<_>>(), vec!["'hello", "'"]); }

    // Tekken: o200k without the contraction suffix, single digits
    #[test] fn tekken_no_contraction() {
        assert_eq!(Tekken::new("I'm").collect::<Vec<_>>(), vec!["I", "'m"]);
        assert_eq!(Tekken::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]);
        assert_eq!(Tekken::new("O'Donnell").collect::<Vec<_>>(), vec!["O", "'Donnell"]);
    }
    #[test] fn tekken_digits() { assert_eq!(Tekken::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn tekken_camelcase() { assert_eq!(Tekken::new("parseJSON").collect::<Vec<_>>(), vec!["parse", "JSON"]); }
    // HF regex parity, from tokbench's multilingual corpora
    #[test] fn camel_upper_excludes_letter_numbers() {
        // `Ⅲ` is Uppercase but Nl (\p{N}), not \p{Lu}
        assert_eq!(O200k::new(" QⅢ").collect::<Vec<_>>(), vec![" Q", "Ⅲ"]);
        assert_eq!(Tekken::new(" QⅢ").collect::<Vec<_>>(), vec![" Q", "Ⅲ"]);
    }
    #[test] fn camel_unicode_punct_prefix_upper() {
        assert_eq!(O200k::new("d’Épinay").collect::<Vec<_>>(), vec!["d", "’Épinay"]);
        assert_eq!(Tekken::new("d’Épinay").collect::<Vec<_>>(), vec!["d", "’Épinay"]);
    }
    #[test] fn camel_apostrophe_prefix_marks() {
        // Lo letter + combining marks after `'` stay one run
        assert_eq!(O200k::new("'जीरो'").collect::<Vec<_>>(), vec!["'जीरो", "'"]);
        assert_eq!(Tekken::new("'ตัว").collect::<Vec<_>>(), vec!["'ตัว"]);
    }
    #[test] fn camel_punct_run_keeps_marks() {
        // ` ?[^\s\p{L}\p{N}]+` takes U+FE0F; a lone symbol + mark is still a letter prefix
        assert_eq!(O200k::new("x ❤\u{FE0F}\n").collect::<Vec<_>>(), vec!["x", " ❤\u{FE0F}\n"]);
        assert_eq!(O200k::new("!!\u{301}x").collect::<Vec<_>>(), vec!["!!\u{301}", "x"]);
        assert_eq!(O200k::new("!\u{301}x").collect::<Vec<_>>(), vec!["!\u{301}x"]);
        assert_eq!(Tekken::new("\u{200D}♀\u{FE0F}").collect::<Vec<_>>(), vec!["\u{200D}♀\u{FE0F}"]);
    }
    #[test] fn nbsp_prefixes_letters() {
        assert_eq!(Cl100k::new(":\u{A0}A b").collect::<Vec<_>>(), vec![":", "\u{A0}A", " b"]);
        assert_eq!(Tekken::new(")\u{A0}Symmetric").collect::<Vec<_>>(), vec![")", "\u{A0}Symmetric"]);
        assert_eq!(Gpt2::new(":\u{A0}A").collect::<Vec<_>>(), vec![":", "\u{A0}", "A"]);
    }
    #[test] fn deepseek_ws_run_before_unicode_numeric() {
        assert_eq!(DeepSeek::new("=\u{A0}\u{A0}½").collect::<Vec<_>>(), vec!["=", "\u{A0}\u{A0}", "½"]);
    }
    #[test] fn tekken_punct_slash() { assert_eq!(Tekken::new("a //\nb").collect::<Vec<_>>(), vec!["a", " //\n", "b"]); }

    // Voyage
    #[test] fn voyage_basic() { assert_eq!(Voyage::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn voyage_single_digits() { assert_eq!(Voyage::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn voyage_contractions() { assert_eq!(Voyage::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn voyage_punct_prefix() { assert_eq!(Voyage::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }

    // SmolLM
    #[test] fn smollm_basic() { assert_eq!(SmolLM::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn smollm_single_digits() { assert_eq!(SmolLM::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn smollm_space_no_digit() {
        assert_eq!(SmolLM::new("test 123").collect::<Vec<_>>(), vec!["test", " ", "1", "2", "3"]);
        assert_eq!(SmolLM::new("a 1 b").collect::<Vec<_>>(), vec!["a", " ", "1", " b"]);
    }
    #[test] fn smollm_contraction() { assert_eq!(SmolLM::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn smollm_whitespace() {
        assert_eq!(SmolLM::new("a\n\nb").collect::<Vec<_>>(), vec!["a", "\n", "\n", "b"]);
        assert_eq!(SmolLM::new("a  b").collect::<Vec<_>>(), vec!["a", " ", " b"]);
    }
    #[test] fn smollm_newline_digits() { assert_eq!(SmolLM::new("abc\n123").collect::<Vec<_>>(), vec!["abc", "\n", "1", "2", "3"]); }
    #[test] fn smollm_whitespace_before_unicode_digit() {
        // \s+(?=\p{N}) exception covers unicode numerics (½ U+00BD), not just ASCII digits
        assert_eq!(SmolLM::new("a\n\n½ cup").collect::<Vec<_>>(), vec!["a", "\n\n", "½", " cup"]);
        assert_eq!(SmolLM::new("x  ½").collect::<Vec<_>>(), vec!["x", "  ", "½"]);
    }

    // DeepSeek
    #[test] fn deepseek_basic() { assert_eq!(DeepSeek::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn deepseek_digits() { assert_eq!(DeepSeek::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn deepseek_contractions() {
        assert_eq!(DeepSeek::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]);
        assert_eq!(DeepSeek::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]);
    }
    #[test] fn deepseek_marks() { assert_eq!(DeepSeek::new("ก\u{0E31}น").collect::<Vec<_>>(), vec!["ก\u{0E31}น"]); }
    #[test] fn deepseek_format_chars_not_punct() {
        // DeepSeek's punct class is [\p{P}\p{S}]: Cf/Cc chars (ZWSP, soft hyphen,
        // C1 controls) are not punct — they act as the letter rule's optional
        // one-char prefix [^\r\n\p{L}\p{P}\p{S}]? or stand alone
        assert_eq!(DeepSeek::new("values \u{200B}\u{200B}that").collect::<Vec<_>>(), vec!["values", " ", "\u{200B}", "\u{200B}that"]);
        assert_eq!(DeepSeek::new("higher \u{AD}partic").collect::<Vec<_>>(), vec!["higher", " ", "\u{AD}partic"]);
        assert_eq!(DeepSeek::new("\u{200B}école").collect::<Vec<_>>(), vec!["\u{200B}école"]);
        assert_eq!(DeepSeek::new("a\u{80}\u{94}b").collect::<Vec<_>>(), vec!["a", "\u{80}", "\u{94}b"]);
        // ASCII controls too: ESC stands alone, then `[` prefixes the letters
        assert_eq!(DeepSeek::new("l\x1b[s\x1b[6n").collect::<Vec<_>>(), vec!["l", "\x1b", "[s", "\x1b", "[", "6", "n"]);
    }
    #[test] fn deepseek_nonascii_punct_no_letter_prefix() {
        // Non-ASCII \p{P}/\p{S} chars are excluded from the letter-prefix class;
        // only the ASCII punct list can prefix letters
        assert_eq!(DeepSeek::new("x «y").collect::<Vec<_>>(), vec!["x", " «", "y"]);
        assert_eq!(DeepSeek::new("«ab").collect::<Vec<_>>(), vec!["«", "ab"]);
    }
    #[test] fn qwen_format_chars_are_punct() {
        // Qwen's negated class [^\s\p{L}\p{N}] DOES include Cf chars
        assert_eq!(Qwen::new("a \u{200B}b").collect::<Vec<_>>(), vec!["a", " \u{200B}", "b"]);
        assert_eq!(Qwen::new("x\u{AD}y").collect::<Vec<_>>(), vec!["x", "\u{AD}y"]);
    }
    #[test] fn indic_marks_complete_table() {
        // Gujarati marks (U+0A81..U+0AFF) were missing from the hand-rolled mark
        // table; now generated complete. DeepSeek merges [\p{L}\p{M}]+ runs;
        // GPT-2's plain \p{L} splits at every mark (marks are punct there);
        // Voyage (Qwen3 dispatch) allows a one-char non-letter prefix.
        assert_eq!(DeepSeek::new("આફ્રિકા ખંડ").collect::<Vec<_>>(), vec!["આફ્રિકા", " ખંડ"]);
        assert_eq!(
            Gpt2::new("આફ્રિકા ખંડ").collect::<Vec<_>>(),
            vec!["આફ", "્", "ર", "િ", "ક", "ા", " ખ", "ં", "ડ"]
        );
        assert_eq!(Voyage::new("આફ્રિકા").collect::<Vec<_>>(), vec!["આફ", "્ર", "િક", "ા"]);
    }
    #[test] fn deepseek_no_contraction_split() {
        // DeepSeek's pattern has no contraction rule; [punct][A-Za-z]+ takes ALL
        // following ASCII letters ("'Toole", "'ts"), and only ASCII ("l", "'", "été")
        assert_eq!(DeepSeek::new("O'Toole").collect::<Vec<_>>(), vec!["O", "'Toole"]);
        assert_eq!(DeepSeek::new("don'ts").collect::<Vec<_>>(), vec!["don", "'ts"]);
        assert_eq!(DeepSeek::new(".\n\n'The").collect::<Vec<_>>(), vec![".\n\n", "'The"]);
        assert_eq!(DeepSeek::new("l'été").collect::<Vec<_>>(), vec!["l", "'", "été"]);
    }
    #[test] fn deepseek_punct_prefix() { assert_eq!(DeepSeek::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }

    // Qwen
    #[test] fn qwen_basic() { assert_eq!(Qwen::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn qwen_single_digits() { assert_eq!(Qwen::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn qwen_contractions() {
        assert_eq!(Qwen::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]);
        assert_eq!(Qwen::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]);
    }
    #[test] fn qwen_marks() { assert_eq!(Qwen::new("ก\u{0E31}น").collect::<Vec<_>>(), vec!["ก\u{0E31}น"]); }
    #[test] fn qwen_punct_prefix() { assert_eq!(Qwen::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }
    #[test] fn qwen_newline() {
        assert_eq!(Qwen::new("a\nb").collect::<Vec<_>>(), vec!["a", "\n", "b"]);
        assert_eq!(Qwen::new("a \nb").collect::<Vec<_>>(), vec!["a", " \n", "b"]);
    }

    // Unicode numerics: \p{N} covers No/Nl (¹ U+00B9, ❶ U+2776), not just ASCII digits
    #[test] fn gpt2_space_unicode_numeric() {
        assert_eq!(Gpt2::new("x ¹").collect::<Vec<_>>(), vec!["x", " ¹"]);
        assert_eq!(Gpt2::new("a ❶b").collect::<Vec<_>>(), vec!["a", " ❶", "b"]);
    }
    #[test] fn gpt2_unicode_numeric_run() {
        assert_eq!(Gpt2::new("¹²³").collect::<Vec<_>>(), vec!["¹²³"]);
        assert_eq!(Gpt2::new("12¹").collect::<Vec<_>>(), vec!["12¹"]);
    }
    #[test] fn cl100k_unicode_numeric_chunked() {
        assert_eq!(Cl100k::new("1¹23").collect::<Vec<_>>(), vec!["1¹2", "3"]);
    }
    #[test] fn qwen_unicode_numeric_single() {
        assert_eq!(Qwen::new("1¹2").collect::<Vec<_>>(), vec!["1", "¹", "2"]);
        assert_eq!(Qwen::new(" ¹").collect::<Vec<_>>(), vec![" ", "¹"]);
    }

    // Multibyte punctuation must consume trailing newlines ([^\s\p{L}\p{N}]+[\r\n]*)
    #[test] fn qwen_multibyte_punct_trailing_newlines() {
        assert_eq!(Qwen::new("smart”\n\nM").collect::<Vec<_>>(), vec!["smart", "”\n\n", "M"]);
        assert_eq!(Qwen::new("you…\nA").collect::<Vec<_>>(), vec!["you", "…\n", "A"]);
        assert_eq!(Qwen::new("”\n \nx").collect::<Vec<_>>(), vec!["”\n", " \n", "x"]);
        assert_eq!(Qwen::new("”…\n\nx").collect::<Vec<_>>(), vec!["”…\n\n", "x"]);
        assert_eq!(Qwen::new("x«\n\ny").collect::<Vec<_>>(), vec!["x", "«\n\n", "y"]);
    }
    #[test] fn cl100k_multibyte_punct_trailing_newlines() {
        assert_eq!(Cl100k::new("a”\n\nb").collect::<Vec<_>>(), vec!["a", "”\n\n", "b"]);
    }
    #[test] fn gpt2_multibyte_punct_no_trailing() {
        // GPT-2 has no [\r\n]* suffix and punct never prefixes letters
        assert_eq!(Gpt2::new("x«\n\ny").collect::<Vec<_>>(), vec!["x", "«", "\n", "\n", "y"]);
        assert_eq!(Gpt2::new("«abc").collect::<Vec<_>>(), vec!["«", "abc"]);
        assert_eq!(Gpt2::new("«»").collect::<Vec<_>>(), vec!["«»"]);
    }

    // Contractions are not blocked by a following letter ('(?i:[sdmt]|ll|ve|re) has no lookahead)
    #[test] fn qwen_contraction_before_letter() {
        assert_eq!(Qwen::new("O'Toole").collect::<Vec<_>>(), vec!["O", "'T", "oole"]);
        assert_eq!(Qwen::new("don'ts").collect::<Vec<_>>(), vec!["don", "'t", "s"]);
        assert_eq!(Qwen::new("'Toole").collect::<Vec<_>>(), vec!["'T", "oole"]);
    }
    #[test] fn gpt2_contraction_before_letter() {
        assert_eq!(Gpt2::new("don'ts").collect::<Vec<_>>(), vec!["don", "'t", "s"]);
    }
    #[test] fn o200k_contraction_suffix_before_letter() {
        assert_eq!(O200k::new("don'ts").collect::<Vec<_>>(), vec!["don't", "s"]);
    }

    // Regressions vs HF (tokbench agentic-swe / japanese / english diffs).
    fn pieces<C: PretokConfig>(s: &str) -> Vec<&str> { Core::<C>::new(s).collect() }
    #[test] fn tab_prefixes_letters_only() {
        // `[^\r\n\p{L}\p{N}]?\p{L}+`: a tab prefixes letters; ` ?[^\s\p{L}\p{N}]+` a space only.
        for got in [pieces::<Cl100kConfig>("8\t_re_num"), pieces::<DeepSeekConfig>("8\t_re_num")] {
            assert_eq!(got, vec!["8", "\t", "_re", "_num"]);
        }
        assert_eq!(pieces::<Cl100kConfig>("\t.re"), vec!["\t", ".re"]);
        assert_eq!(pieces::<Cl100kConfig>("\t8"), vec!["\t", "8"]);
        assert_eq!(pieces::<Cl100kConfig>("\tre"), vec!["\tre"]);
        assert_eq!(pieces::<QwenConfig>("\t_re"), vec!["\t", "_re"]);
        assert_eq!(pieces::<O200kConfig>("\t_re"), vec!["\t", "_re"]);
        // GPT-2 ` ?\p{L}+`: a tab prefixes nothing.
        assert_eq!(pieces::<Gpt2Config>("8\t_re_num"), vec!["8", "\t", "_", "re", "_", "num"]);
        assert_eq!(pieces::<Gpt2Config>("\tre"), vec!["\t", "re"]);
    }
    #[test] fn cl100k_newline_run_through_last_newline() {
        // `\s*[\r\n]+` reaches the last newline of the whole whitespace run.
        assert_eq!(pieces::<Cl100kConfig>("\r\n  \r\n  x"), vec!["\r\n  \r\n", " ", " x"]);
        assert_eq!(pieces::<DeepSeekConfig>("\r\n  \r\n  x"), vec!["\r\n  \r\n", " ", " x"]);
        assert_eq!(pieces::<Cl100kConfig>("\n \n"), vec!["\n \n"]);
        assert_eq!(pieces::<Gpt2Config>("\r\n  \r\n  x"), vec!["\r\n  \r\n ", " x"]);
    }
    #[test] fn letter_numbers_are_not_letters() {
        // Nl (Ⅲ) is Alphabetic but \p{N}, not \p{L}.
        assert_eq!(pieces::<Cl100kConfig>("aⅢ"), vec!["a", "Ⅲ"]);
        assert_eq!(pieces::<Cl100kConfig>(" Ⅲ"), vec![" ", "Ⅲ"]);
        assert_eq!(pieces::<DeepSeekConfig>(" Ⅲ"), vec![" ", "Ⅲ"]);
        assert_eq!(pieces::<Gpt2Config>("aⅢ"), vec!["a", "Ⅲ"]);
        assert_eq!(pieces::<Gpt2Config>(" Ⅲ"), vec![" Ⅲ"]);
    }
    #[test] fn deepseek_cjk_runs_are_segments() {
        // `[一-龥぀-ゟ゠-ヿ]+` is split off before the main regex, which then
        // re-splits each segment.
        assert_eq!(
            pieces::<DeepSeekConfig>("（BD-ROM・-R・-RE）"),
            vec!["（", "BD", "-ROM", "・", "-R", "・", "-RE", "）"]
        );
        assert_eq!(pieces::<DeepSeekConfig>("Ｆ一"), vec!["Ｆ", "一"]);
        assert_eq!(pieces::<DeepSeekConfig>(" 中文"), vec![" ", "中文"]);
        assert_eq!(pieces::<DeepSeekConfig>("ア゛b"), vec!["ア", "゛", "b"]);
        assert_eq!(pieces::<DeepSeekConfig>("゛゛"), vec!["゛゛"]);
        assert_eq!(pieces::<DeepSeekConfig>("-中"), vec!["-", "中"]);
    }
    #[test] fn deepseek_ascii_punct_prefixes_ascii_letters_only() {
        // `[!-/:-@[-`{-~][A-Za-z]+`; `[^\r\n\p{L}\p{P}\p{S}]?` excludes punct.
        assert_eq!(pieces::<DeepSeekConfig>("=σλ"), vec!["=", "σλ"]);
        assert_eq!(pieces::<DeepSeekConfig>("-handâa"), vec!["-hand", "âa"]);
    }

    // HF-oracle parity (tests/golden.rs, `--features oracle`).
    #[test] fn space_only_apostrophe_is_punct() {
        // GPT-2 ` ?[^\s\p{L}\p{N}]+`: a non-contraction apostrophe never prefixes letters.
        assert_eq!(pieces::<Gpt2Config>("d'Arce o'clock"), vec!["d", "'", "Arce", " o", "'", "clock"]);
        assert_eq!(pieces::<Gpt2Config>("DON'T"), vec!["DON", "'", "T"]);
        assert_eq!(pieces::<Gpt2Config>("\n\n'We"), vec!["\n", "\n", "'", "We"]);
        assert_eq!(pieces::<SmolLMConfig>("d'Andorra"), vec!["d", "'", "Andorra"]);
        // Cl100k-family still takes it as a one-char letter prefix.
        assert_eq!(pieces::<Cl100kConfig>("d'Arce"), vec!["d", "'Arce"]);
    }
    #[test] fn camel_lo_lm_marks_are_both_cases() {
        // `U*L+|U+L*` with Lm/Lo/M in both U and L: U* backtracks to the last
        // L-class char when no Ll follows.
        assert_eq!(pieces::<O200kConfig>("한Q"), vec!["한", "Q"]);
        assert_eq!(pieces::<O200kConfig>(":惡搞RFC"), vec![":惡搞", "RFC"]);
        assert_eq!(pieces::<O200kConfig>("\u{301}Q_"), vec!["\u{301}", "Q", "_"]);
        assert_eq!(pieces::<O200kConfig>("+\u{301}Жl"), vec!["+\u{301}Жl"]);
        assert_eq!(pieces::<O200kConfig>("Q'\u{301}Xß中"), vec!["Q", "'\u{301}Xß中"]);
        assert_eq!(pieces::<TekkenConfig>("\u{a0}한Q"), vec!["\u{a0}한", "Q"]);
    }
    #[test] fn deepseek_unicode_ws_prefixes_letters() {
        // `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`: NBSP/U+2003/U+3000 are valid prefixes.
        assert_eq!(pieces::<DeepSeekConfig>("80\u{a0}km"), vec!["80", "\u{a0}km"]);
        assert_eq!(pieces::<DeepSeekConfig>("a\u{3000}b  c"), vec!["a", "\u{3000}b", " ", " c"]);
        assert_eq!(pieces::<DeepSeekConfig>("z\u{2003}e"), vec!["z", "\u{2003}e"]);
        // The prefix takes the full letter run, not the ASCII-only punct rule's.
        assert_eq!(pieces::<DeepSeekConfig>("\u{a0}héllo"), vec!["\u{a0}héllo"]);
    }
}
