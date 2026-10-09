# pretokie v2 / Phase 0 — HF tokenizers engine investigation

Reference: local clone of `huggingface/tokenizers` at `~/Workspaces/tokenizers`,
commit `d582781` ("Add a ParityBpeTrainer example…", post-0.21.x main).
Everything below cites that tree (`tokenizers/src/...`).

This doc is the **spec source of truth** for the pretokie v2 rewrite: the
regex engine HF uses, its exact compile configuration, the canonical
pre-tokenization pipeline per scheme, and the semantic traps our
implementation and tests must honor.

## 1. The engine

**Default engine: Oniguruma, via the `onig` crate (Cargo.toml pins `6.5.1`,
`default-features = false`).** `unstable_wasm` swaps in `fancy-regex`
(known semantic differences — never our reference).

The wrapper is `utils/onig.rs`: `SysRegex::new(pattern)` → plain
`onig::Regex::new(pattern)`. No explicit options anywhere, so the onig-crate
defaults apply, verified against docs.rs/onig:

- syntax: `ONIG_SYNTAX_RUBY`
- options: `ONIG_OPTION_NONE` (not even SINGLELINE)
- encoding: UTF-8

An oracle that compiles patterns with `onig::Regex::new` is therefore
**bit-identical to HF's compilation**. Nothing else to match.

### 1.1 How a `SysRegex` becomes pretokens

`Pattern for &Regex::find_matches` (`tokenizer/../utils/onig.rs:25-44`):

```
for (start, end) in regex.find_iter(text):
    if prev != start: splits.push(((prev, start), false))   # gap
    splits.push(((start, end), true))                       # match
if prev != len: splits.push(((prev, len), false))           # trailing gap
```

Then `PreTokenizedString::split(pattern, behavior)` applies a
`SplitDelimiterBehavior` to the `(span, matched)` list. The two behaviors
that matter for our schemes:

- **`Isolated`**: every matched span becomes its own pretoken; gaps are kept
  as their own pretokens (zero-length spans don't produce pieces).
- **`Removed`**: matched spans are dropped; unmatched spans are kept.
- **`Invert(&re)`** (from `tokenizer/pattern.rs`): flips matched↔gap in the
  find_matches output before behavior is applied.

So there are exactly two canonical shapes used by the schemes we support:

| shape | stages | effect |
|---|---|---|
| ByteLevel (`use_regex=true`) | split on the ByteLevel `RE`, `Isolated` | pretokens = regex matches (coverage is total → no gaps) |
| Split(regex, **Removed, invert:true**) + ByteLevel(`use_regex=false`) | pretokens = regex matches; inter-match gaps (typically empty) removed |

`Sequence` applies stages in order; each stage re-splits the pieces of the
previous one (`pre_tokenizers/sequence.rs`).

### 1.2 Second engine in the crate (do not confuse)

`pre_tokenizers/whitespace.rs` uses the **Rust `regex` crate**
(`\w+|[^\w\s]+`, Invert+Removed) — not onig. Irrelevant for our schemes
(BERT uses its own char-class splitter, §2), but a trap when grepping the
crate for "the" regex engine.

### 1.3 Oniguruma semantic traps to honor (and test empirically)

The oracle must reproduce these, and our schemes must be tested against
them — they are exactly where a Rust-regex mental model goes wrong:

1. **`\s` under RUBY syntax + UTF-8 — SETTLED EMPIRICALLY (oracle pins,
   `tests/golden.rs::onig_semantics`): Oniguruma's `\s` IS Unicode-aware**
   — it matches NBSP (U+00A0), U+2003, U+2028, U+3000 (and `\S` excludes
   them). `\w` is likewise Unicode-aware (`\w` matches `é`, `中`). This
   overturned the doc's original ASCII-only hypothesis, and it has real
   consequences: for cl100k-family patterns, `\u{a0}` both joins `\s*[\r\n]+`
   runs and is a valid `[^\r\n\p{L}\p{N}]?` word prefix. Any class table
   built from ASCII-only whitespace will diverge.
2. **`\p{L}` etc.**: Oniguruma ships its own Unicode property tables
   (bundled UCD version) — possibly different Unicode version than
   ICU (what gigatoken/pretokie use) or Rust `regex-syntax`. Any char
   whose category changed between versions is a potential divergence.
3. **`(?i:...)` case folding**: verified working (simple, unicode-aware);
   `'T` matches `(?i:'t)`, `'Ll` does not.
4. **Lookaheads** `(?!\S)`: verified working, including the greedy
   backtrack (`\s+(?!\S)` on `"  x"` matches 1 space) and zero-width
   end-of-string matches. Rust `regex` cannot express these (why
   pretokie's `Regex` fallback hand-rewrites `\s+(?!\S)` → `\s+$` + trim,
   and why fancy-regex was the wasm choice).
5. **Possessive quantifiers** (`\p{L}++`, `\s++$`): supported by Oniguruma
   (Ruby syntax). tiktoken's `openai_public.py` uses them as an
   optimization; **HF's patterns don't**. Segmentation is equivalent for
   these patterns, but the oracle and all goldens use **HF's exact
   strings** (§2), never tiktoken's.
6. **`[:punct:]`-style POSIX classes**: Oniguruma's definitions have
   historically differed from Rust's (`\p{P}` vs `\p{P}\p{S}` — see HF
   issue #1057). Our patterns use explicit `\p{P}`/`\p{S}` mostly, but
   BERT's punctuation rule is char-class code, not regex (§2).
7. **Zero-width matches**: `find_matches` skips zero-length gaps
   (`prev != start` guard) but a zero-width *match* would be pushed as
   `((i,i),true)`. All our patterns require ≥1 char, but the oracle
   should assert this invariant per pattern.

## 2. Canonical per-scheme pipelines (spec)

Verbatim pattern strings as they must appear in the oracle and goldens.
Sources: crate-internal constants for ByteLevel; real `tokenizer.json`
`pre_tokenizer` sections (fetched from the Hub) for Split-based schemes.
tiktoken's `openai_public.py` cross-checked but **not** used verbatim
(possessive-quantifier variants, `\s` vs `\s+` endings).

### r50k / gpt2 — `ByteLevel`

Crate-internal (`pre_tokenizers/byte_level.rs:43-46`):

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Pipeline: one stage — split on `RE`, `Isolated`. Config flags from
tokenizer.json that affect boundaries:

- `add_prefix_space` (gpt2: **false**; many other ByteLevel models: true →
  a space is *prepended to the whole input* before splitting — affects the
  first pretoken). NOTE: tokie's hf.rs detection ignores this flag today —
  recorded landmine.
- `use_regex` (gpt2: true; Split-based schemes: false).

The ByteLevel byte↔unicode remap happens *after* splitting (a
normalization on each piece) and does not affect pretoken boundaries.

Known tokie divergence (this rewrite's bug A/B): HF `d'Arce` →
`d`, `'`, `Arce`; `a\tb` → `a`, `\t`, `b`.

### cl100k — `Split(Removed, invert)` + ByteLevel

Xenova/gpt-4 (canonical cl100k_base JSON):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Pipeline: `Split(RE, Removed, invert: true)` → `ByteLevel(add_prefix_space=false, use_regex=false)`.
Digits chunked in 3s; contraction alternative is **case-insensitive**;
letter run may be prefixed by ONE non-letter-non-digit that is not `\r\n`
(this is where tokie's `PunctPrefixMode::Any` comes from — correct here,
wrong when reused for gpt2); ` ?[^\s\p{L}\p{N}]+[\r\n]*` — punct runs
*trailing newlines* attach (PunctTrailing::Newlines).

### o200k — `Split(Removed, invert)` + ByteLevel

Xenova/gpt-4o (7 alternatives, CamelCase-aware):

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Pipeline: same shape as cl100k. Distinctives: explicit
`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` vs `[\p{Ll}…]` case classes (CamelCase
splitting — tokie's `CamelCase` LETTER_MODE); contraction is an optional
*suffix* on the word alternatives (ContractionMode::Suffix); punct runs
attach trailing `[\r\n/]*` — **newlines and slashes** (PunctTrailing::
NewlinesAndSlashes).

### qwen2 — `Split(Isolated)` + ByteLevel

Qwen/Qwen2-7B:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Pipeline: `Split(RE, Isolated, invert: false)` → ByteLevel(`use_regex=false`).
cl100k-family but **single digits** (`\p{N}`) and `trim_offsets: false`.
tokie's `QwenConfig`: marks `[\p{L}\p{M}]+`? — note Qwen2's pattern has NO
`\p{M}`; Qwen3.5's does (tokie's Qwen35 scheme). Grab Qwen3.5's JSON when
building its goldens.

### deepseek — 3× `Split(Isolated)` + ByteLevel

deepseek-ai/DeepSeek-V3 — **stage order is part of the spec**:

1. `\p{N}{1,3}` (Isolated)
2. `[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+` (Isolated) — CJK + Kana
   ranges (literal ranges, not categories)
3. ```
   [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   ```
   (Isolated)

Distinctives: ASCII-only explicit punctuation-prefix-letters rule
(tokie's DeepSeek "ASCII-only absorb"); letters include marks
(`[\p{L}\p{M}]+`); punct class is `\p{P}\p{S}`; digits already isolated by
stage 1 (Chunked3). `PunctTrailing::Newlines` on the punct stage.

### smollm — `Digits(individual)` + ByteLevel(use_regex=true)

HuggingFaceTB/SmolLM2-1.7B:

1. `Digits { individual_digits: true }` — `char::is_numeric` split, Isolated
2. `ByteLevel { add_prefix_space: false, use_regex: true }` — the r50k RE

So: **every numeric char isolated first**, then the gpt2 regex runs on the
remaining pieces (and cannot merge digits back). `char::is_numeric` here =
Rust's `is_numeric` (Unicode N* — includes ½, Ⅻ; matches tokie's
DigitMode::Single semantics).

### bert — char-class code, no regex

`pre_tokenizers/bert.rs`: two stages inside `BertPreTokenizer`:

1. split on `char::is_whitespace` — **Removed** (whitespace deleted, so BERT
   pieces never contain whitespace)
2. split on `is_bert_punc = char::is_ascii_punctuation(c) || c.is_punctuation()`
   — **Isolated**; `is_punctuation` is the `unicode_categories` crate's
   Unicode **P** category only.

Trap: ASCII `$+<>=^|~` are ASCII punctuation (split), but **non-ASCII
symbols (€ © → ±) are Unicode S, not P → not split** by BERT. tokie's
Bert impl must match this exactly, not use a general `\p{P}\p{S}` class.

### voyage — RESOLVED (voyageai/voyage-3, published on HF)

All Voyage models on the Hub (`voyage-3`, `voyage-3-lite`, `voyage-code-3`
— checked all three, identical):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Pipeline: `Split(RE, Isolated, invert: false)` → ByteLevel(`use_regex=false`)
— byte-for-byte the **Qwen2 shape** (cl100k-family, single digits),
plus an `NFC` normalizer upstream (normalization precedes pretokenization;
doesn't change the scheme's boundary rules). tokie's `VoyageConfig`
matches this. Voyage edge cases + goldens + fuzz are now in the suite.

### regex-fallback family (e.g. Falcon) — multi-stage generic

tiiuae/falcon-7b: `Punctuation(Contiguous)` → `ByteLevel(use_regex=true)` →
`Digits(individual=false → Contiguous)` → `Split("[0-9][0-9][0-9]", Isolated)`.
tokie serves these via `Pretokenizer::Regex` (regex-automata fallback with
the `\s+(?!\S)` rewrite). The oracle must be able to express these
sequences too (Punctuation/Digits stages are char-class code, not regex).

### metaspace / none (SentencePiece)

`Metaspace` (pre_tokenizers/metaspace.rs) is normalization-driven; tokie
maps it (and unrecognized pre_tokenizers) to `PretokType::None` — no
pretokenization. Out of oracle scope except as a "no-op" case.

## 3. Implications for the rewrite (recorded now, resolved in Phases 1-2)

1. **Stages, not single patterns.** Half our schemes are *sequences*
   (Split→ByteLevel, Digits→ByteLevel, 3×Split). The new `Scheme` trait's
   scalar `advance` must implement the *composed* pipeline, and the SIMD
   `batch_masks` must handle stage interactions (e.g. smollm: digit
   isolation happens *before* the regex ever sees the text — so the mask
   scanner must isolate numeric chars individually, then apply r50k rules
   to the rest; that's tokie's SmolLM digit exception today).
2. **ByteLevel flags are scheme inputs.** `add_prefix_space`/`use_regex`
   from tokenizer.json must be threaded through detection into the scheme
   (today they're dropped — the add_prefix_space landmine above).
3. **Behavior shapes are equivalent for total-coverage patterns**, but the
   oracle implements Isolated/Removed/Invert faithfully so goldens are
   generated by the same code path HF uses, not an equivalence assumption.
4. **Unicode table skew is a live risk**: onig (own UCD) vs ICU (gigatoken)
   vs unicode-general-category (pretokie) vs Rust std
   (`char::is_numeric`/`is_whitespace` in Digits/BERT stages — std, not
   ICU!). Our packed class tables must be validated against onig for
   `\p{...}` boundaries, and against Rust std for the char-class stages.
5. **`\s` definition** (trap 1.3.1) is the single highest-impact unknown —
   it appears in every pattern. First oracle test: pin it down empirically
   against Oniguruma.

## 4. Oracle — BUILT (option A), validated 2026-08-27

Implemented as an **opt-in `oracle` feature of pretokie** (no new crate):

- `crates/pretokie/src/oracle.rs` — the engine: verbatim patterns,
  find_matches/behaviors/Invert/Sequence, char-class stages, and the
  ByteLevel remap as a mid-sequence transform (post-remap stages operate
  on remapped text — Falcon's Digits splits `¹` (byte 0xB9) out
  mid-character, exactly like HF).
- `crates/pretokie/tests/golden.rs` — three layers: `onig_semantics` pins
  (always on), per-scheme edge-case goldens, data goldens
  (`benches/data/enwik8`, `benches/data/owt_sample.txt`, 1MB chunks,
  `PRETOKIE_GOLDEN_BYTES` env cap, files skipped when absent), and a
  deterministic fuzz differential (`PRETOKIE_FUZZ_ITERS`).
- `crates/pretokie/examples/dump.rs` — chunked dump
  (`--scheme X --impl oracle|pretokie --file F|--text T`).
- `scripts/verify_oracle.py` — one-shot loop-closure proof: oracle dump vs
  Python `tokenizers` `pre_tokenize_str` for all 8 schemes.

**Validation result: oracle == HuggingFace, bit-exact, all 9 schemes**
(r50k, cl100k, o200k, qwen2, voyage, deepseek, smollm, bert, falcon;
32-case suite: `uv run --with tokenizers --with huggingface_hub
python scripts/verify_oracle.py`).

Run the suite: `cargo test -p pretokie --features oracle --lib --tests`
(needs a C toolchain for onig — nix: `nix shell nixpkgs#cargo
nixpkgs#clang`; MSVC on Windows). Knobs: `PRETOKIE_FUZZ_ITERS`,
`PRETOKIE_GOLDEN_BYTES` (0 = whole file), `PRETOKIE_ISA=scalar|sse2|avx2`.

## 4b. Parity status (dev after 133c164 + oracle fixes)

**All schemes match the oracle; nothing is `#[ignore]`d.** Coverage: edge
cases, enwik8 + OWT data chunks, a short-string fuzz (scalar `Core` path)
and a long mostly-ASCII fuzz (SIMD `Mask` batch path), under every
`PRETOKIE_ISA` backend. The alphabet includes the §1.3 traps (VT/FF/NEL,
Lt/Lm/Lo letters, Other_Alphabetic symbols, Cf/Cc, Ps/Pe).

On `dev` before the fixes, cl100k and voyage already passed everything
(tab prefix, NBSP prefix and `\s*[\r\n]+` runs had been fixed since
v0.1.3). The remaining divergence classes and their fixes:

| class | schemes | HF | fix |
|---|---|---|---|
| apostrophe absorbed into letters | r50k, smollm | `d'Arce` → `d` `'` `Arce` | SpaceOnly configs send a non-contraction `'` to the punct run (`Core` + mask fixup) |
| CamelCase with Lo/Lm/M | o200k (+Tekken) | `한Q` → `한` `Q`; `+́Жl` one piece | `scan_camel`: exact `U*L+\|U+L*` backtracking with Lo/Lm/M in both classes, by general category |
| unicode-ws letter prefix | deepseek | `80\u{a0}km` → `80` `\u{a0}km` | `[^\r\n\p{L}\p{P}\p{S}]?` admits NBSP/U+2003/U+3000 too |
| unmatched gap runs | deepseek | `\x1b\x1b` one piece | `Isolated` keeps a run of unmatched Cc/Cf chars together, except a last one that prefixes letters |
| VT/FF are `\s` | all regex schemes | `a\x0bb` → like `a\tb` | `Core` treats 0x0B/0x0C as tab; mask tab lane widened to 0x09..0x0C minus `\n` |
| Other_Alphabetic symbols | all regex schemes | `Ⓐ` is `\p{S}` | `is_alpha` excludes U+24B6..24E9 and U+1F130..1F189 |
| CJK split | bert | `BertPreTokenizer` keeps `中文` | CJK isolation moved to tokie's BertNormalizer (`handle_chinese_chars`); ASCII controls stay in words, VT/FF are whitespace |
| harness: qwen2 mapping | qwen2 | — | qwen2 goldens run `pretokie::Voyage` (what tokie uses for Qwen2's pattern); `pretokie::Qwen` is Qwen3.5 |
| oracle: BERT punct | bert, falcon | `unicode_categories::is_punctuation` includes Ps | oracle's `is_bert_punc` now includes OpenPunctuation |

Known remaining gaps (not covered by any golden): Unicode-version skew
between onig's tables and Rust std / `unicode-general-category` for
recently assigned code points; `handle_chinese_chars=false` is not
representable in tokie's `Normalizer` (CJK is always padded for BERT
normalizers, as pretokie's per-char split did before).

### Historical baseline (pretokie v0.1.3 vs the oracle)

| scheme | edge cases | real data | dominant divergence classes |
|---|---|---|---|
| r50k | 4/32 fail | enwik8 + OWT chunk 1 | apostrophe absorbed into letters (`'clock`), ws word-prefix |
| cl100k | 3/37 | enwik8 chunk 1 | unicode-ws word prefix (NBSP), `\s*[\r\n]+` runs split wrong |
| o200k | 3/37 | enwik8 chunk 1 | same ws classes as cl100k |
| qwen2 | 4/36 | enwik8 chunk 1 | same ws classes as cl100k |
| voyage | 3/37 | enwik8 chunk 1 | same ws classes as cl100k (pipeline = Qwen2 shape) |
| deepseek | 8/38 | enwik8 chunk 1 | CJK/Kana literal ranges, ws runs |
| smollm | 5/36 | enwik8 chunk 1 | r50k apostrophe + ws classes |
| bert | 3/37 | enwik8 chunk 1 | fast `BertPreTokenizer` does NOT split CJK (pretokie splits per char, the Python-BERT behavior) |
| falcon | n/a (oracle-only) | n/a | no hand-written impl in pretokie (tokie uses regex fallback) |

## 5. Canonical strings file

The verbatim patterns live in `crates/pretokie/src/oracle.rs` (§4) — each
with its pipeline shape (stage list + behavior).
