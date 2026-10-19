# Changelog

## Unreleased

### Performance

- **Unicode classification is one table read.** A generated per-code-point
  class table (`src/unicode_class.rs`, two levels, about 45 KB) replaces the
  std `is_alphabetic`/`is_numeric` skip-list searches and the mark range
  binary search behind the `util` predicates. Output is unchanged: the table
  is generated from the previous predicate implementations and checked
  against them for every code point. The `util` predicates drop from
  20–220 ns to about 1 ns per non-ASCII char, and pretokenization of
  non-Latin text is about 9–15× faster (`Mask` and `Core`, geomean over 13
  scripts); Latin text gains about 7–22%.
- `Core` classifies each char once: non-ASCII letter runs no longer re-enter
  the SWAR ASCII probe per char, and a letter already checked after a space,
  apostrophe or punct prefix is not decoded again.

## 0.2.0

### Breaking changes

- **`Bert` no longer isolates CJK ideographs.** It now matches HuggingFace's
  `BertPreTokenizer` exactly; HF isolates CJK in `BertNormalizer`
  (`handle_chinese_chars`), and tokie's BERT normalizers now do the same.
  Non-whitespace control characters also stay inside words instead of
  splitting them, and VT/FF are whitespace.
  *Migration:* if you call `pretokie::Bert` directly on un-normalized text,
  pad CJK chars with spaces first (or use tokie's BERT normalizer).
- **`util::is_unicode_letter`** now returns `false` for letter numbers
  (Nl: `Ⅲ`, `ⅰ`, `〇`, which are `\p{N}`) and Other_Alphabetic symbols
  (circled `Ⓐ`..`ⓩ`, U+1F130..U+1F189, which are `\p{So}`).
- **`util::is_punct_or_symbol`** now returns `true` for those
  Other_Alphabetic symbols.

### Added

- `Tekken` scheme (`TekkenConfig`): Mistral Tekken (Mistral Nemo, Nemotron)
  — o200k's case-aware letter runs, no contraction suffix, single digits.
- x86_64 SIMD `Mask` path: SSE2 by default, runtime-detected AVX2 opt-in;
  `PRETOKIE_ISA=scalar|base|sse2|neon|avx2` overrides the backend (see
  README).
- Dev-only `oracle` feature (off by default, hidden from docs): an
  Oniguruma-backed re-implementation of HF pretokenization, used by the
  golden tests.

### Fixed (HF-oracle parity)

Every scheme now matches HuggingFace `tokenizers` on edge cases, fuzzing and
enwik8/OWT data:

- r50k/SmolLM: a non-contraction apostrophe is punctuation, never a letter
  prefix (`d'Arce` → `d` `'` `Arce`).
- o200k/Tekken: exact `U*L+|U+L*` CamelCase backtracking with Lo/Lm/M in
  both case classes (`한Q` → `한` `Q`).
- DeepSeek: Unicode whitespace prefixes letters; runs of unmatched Cc/Cf
  chars stay one piece.
- All regex schemes: VT/FF are whitespace (Oniguruma `\s`); Other_Alphabetic
  symbols (circled letters) are not letters.
- Bert: `BertPreTokenizer` parity (see Breaking changes).
