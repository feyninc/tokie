# Bulk-Throughput Gap vs Gigatoken — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (executed inline this session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close tokie's bulk-throughput gap vs gigatoken's ~1.9 GB/s flat-buffer path: mask-scanner pretokenizer (core, biggest lever), flat-buffer batch API (output contract), persistent worker pool (cache reuse); plus the SP added-token lstrip/rstrip parity bug.

**Architecture:** Port gigatoken's mask-scanner design (64-byte NEON classification → u64 boundary algebra → bit-popping walker with scalar-fallback "bad zones") into pretokie as a *generic* `Mask<C: PretokConfig>` layered on the existing `Core<C>` scalar iterator, which remains the ground truth for bad zones, tails, and non-aarch64. v1 keeps ALL non-ASCII, apostrophe-contraction, and config-hard cases in bad zones (scalar re-derived) — parity is structural, not hoped-for.

**Tech Stack:** Rust (NEON intrinsics on aarch64, scalar elsewhere), pyo3 + rust-numpy for flat output.

## Global Constraints

- Worktree: `~/Workspace/tokie-perf`, branch `perf/hotpath`. No Claude attribution in commits.
- Parity gates: `cargo test -p pretokie`, `cargo run --release --example owt_compare --features build -- tokiers/gpt2 openai-community/gpt2` and `-- tokiers/DeepSeek-V3 deepseek-ai/DeepSeek-V3` must report 20342/20342.
- Final gate: `cargo test -p tokie --test accuracy --features build -- --ignored`.
- Benchmarks: `TOKIE_JSON=<gpt2 tokenizer.json> cargo run --release -p tokie --features build --example profile_hotpath`; record baseline BEFORE task 2.
- Coordinate with any active charsmap/serde session before touching .tkz-adjacent code (check before Task 5; Tasks 2–4 do not touch serde).

---

### Task 1: Baseline measurements

- [ ] Locate/fetch a local gpt2 tokenizer.json (HF cache or download once), record `TOKIE_JSON` path.
- [ ] Run profile_hotpath 3×, record pretokenize-only, count 1T, encode 1T, count-batch MT MB/s.
- [ ] Run gigatoken's comparable numbers only if already recorded (do not invest time re-benching gigatoken).

### Task 2: Mask-scanner pretokenizer (pretokie)

**Files:**
- Create: `crates/pretokie/src/core/mask.rs` (MaskState walker + NEON classify + generic algebra)
- Modify: `crates/pretokie/src/core/mod.rs` (add `pub mod mask;`)
- Modify: `crates/pretokie/src/core/iter.rs` (add `Core::with_pos` + `Core::pos` accessors so mask path can use Core as scalar `advance`)
- Modify: `crates/pretokie/src/lib.rs` (swap public aliases `Gpt2 = Mask<'a, Gpt2Config>` etc. per-config as parity lands; keep `Core` exported)
- Test: differential tests inside `mask.rs` (unit vectors from iter.rs tests + randomized fuzz vs `Core<C>` + owt_sample differential)

**Interfaces:**
- Produces: `pub struct Mask<'a, C: PretokConfig>` implementing `Iterator<Item = &'a str>` with `new(&str)`, byte-identical output to `Core<C>`.

**Design (v1):**
1. NEON 64-byte classify → `AsciiMasks { l, d, s, wt, n, hi, ap }` + simdjson movemask64 (port from gigatoken r50k/mask.rs, adapted: tokie's ws set is exactly `{' ','\t','\n','\r'}` to match `Core`, NOT gigatoken's 9..=12).
2. Boundary algebra parameterized by `C`:
   - cont_same for letters/digits/punct (`o = !(l|d|ws|hi)`), `after_sp` suppression applied to letters + punct (+ digits iff `SPACE_PREFIXES_DIGITS`).
   - CamelCase (O200k): extra boundary `upper & (lower << 1 | p_lower)` inside ASCII letter runs.
   - DigitMode: Unlimited → cont_same; Chunked3 → `digit_run_splits3` (+ leading-run bad when a digit run carries in from the previous batch, + trailing digit-run-adjacent-to-hi bad); Single → every digit a boundary.
   - Ws: Gpt2 pattern → run-start + split-before-last-char (`split_ok`) with bit-63 lookahead; WsException::Digits → mask split before ASCII digit, defer bit-63 non-ASCII lookahead via decode; Cl100k pattern → pure-nl runs one piece, pure-non-nl runs Gpt2-style, **mixed nl/space runs → bad zone** (v1), WsException::Cjk only reachable at non-ASCII → bad.
   - PunctTrailing (Newlines / NewlinesAndSlashes): propagate trailing `[\r\n/]` absorption after punct runs via log-doubling; exclude absorbed nl from effective-ws for run-start bits.
   - Punct-prefix absorb (Any/AsciiOnly): `boundary &= !(((o & boundary) << 1) & l)`; punct at bit 63 → bad bit 63.
   - Contractions (`CONTRACTION_MODE != None`): bad-smear `ap | ap<<1 | ap<<2 | ap<<3`, apostrophe at bits ≥ 61 → `bad |= MAX << i`.
   - Non-ASCII: no in-mask classification in v1 — `bad |= hi | hi<<1 | hi>>1` (+ digit-run extension for Chunked3); prev byte at `scan-1` ≥ 0x80 → bad bits 0..2. DeepSeek ASCII controls (not in `[\p{P}\p{S}]`) → bad smear.
   - Carries at bit 0 from `bytes[scan-1]` (ASCII only; hi handled above).
3. `MaskState` walker: direct port of gigatoken's (pos/scan/mask_base/rem/batch_usable/batch_bad/scalar_until + one-batch-ahead precompute + grid-preserving overrun). Scalar advance = `Core::<C>::with_pos(bytes, pos).next()` end position.
4. Non-aarch64: `scalar_until = usize::MAX` (pure Core behavior).

**Steps:**
- [ ] Port movemask64 + ascii_masks (NEON) into `mask.rs`; unit-test masks against scalar predicates on random bytes.
- [ ] Implement `MaskState` + `Mask<C>` with `batch_masks` returning `(0, u64::MAX)` (all-scalar); differential test vs Core on owt_sample for all 7 configs — must pass trivially.
- [ ] Implement Gpt2Config algebra; differential fuzz (random ASCII + mixed unicode strings, 1e6 cases) + owt_sample differential; swap `Gpt2` alias; run owt_compare gpt2 (20342/20342) + profile_hotpath (expect pretokenize-only ≥ 800 MB/s).
- [ ] Extend algebra per config family: Cl100k, Voyage, Qwen, SmolLM, DeepSeek, O200k — each: differential fuzz + owt differential, swap alias, commit.
- [ ] owt_compare DeepSeek-V3 (20342/20342).
- [ ] Commit per family; final commit swaps remaining aliases.

### Task 3: Flat-buffer batch API

**Files:**
- Modify: `crates/tokie/src/tokenizer.rs` (add `encode_batch_flat`)
- Modify: `crates/tokie-python/src/lib.rs` + `crates/tokie-python/Cargo.toml` (numpy dep, `encode_batch_flat` → `(np.uint32 ids, np.uint64 lengths)`)

**Interfaces:**
- Rust: `pub fn encode_batch_flat(&self, texts: &[&str], add_special_tokens: bool) -> (Vec<u32>, Vec<u64>)` — flat ids + per-doc lengths, parallel chunks concatenated in order.
- Python: `encode_batch_flat(texts, add_special_tokens=True) -> tuple[np.ndarray, np.ndarray]` via `IntoPyArray` (zero-copy Vec handoff), computed under `allow_threads`.

**Steps:**
- [ ] Rust core + unit test (flat == concat of encode_batch ids; lengths match).
- [ ] Python binding + smoke test via maturin/venv if available; commit.

### Task 4: Persistent worker pool

**Files:**
- Create: `crates/tokie/src/pool.rs` (global lazy pool: N = available_parallelism workers, crossbeam-free std mpsc jobs)
- Modify: `crates/tokie/src/tokenizer.rs` (encode_batch / count_tokens_batch / encode_batch_flat submit to pool)

**Design:** Each worker owns a lazily-built `PretokenCache` that persists across calls. Cache is tokenizer-specific → tag each worker cache with a generation id (per-Tokenizer monotonic id assigned at construction); on mismatch, clear/rebuild. Scoped lifetime: jobs borrow `&Tokenizer` — use `std::thread::scope`-style API replaced by pool: send erased closures with a completion channel; unsafe lifetime extension confined to one audited site (or make jobs `'static` by passing Arc'd inputs — decide at implementation; simplest correct: keep scoped threads but move cache ownership to a global per-thread-slot cache pool, `Vec<Mutex<Option<PretokenCache>>>`, workers check one out per call — avoids rebuild cost without pool plumbing).
- [ ] Implement cache checkout pool (or full worker pool if cleaner); benchmark encode_batch before/after; commit.

### Task 5: SP added-token lstrip/rstrip bug

- [ ] Check active sessions for charsmap/serde work before touching anything .tkz-adjacent.
- [ ] Repro: TinyLlama "Hello </s> world" vs HF (script under scratchpad, HF `tokenizers` via python venv or the owt_compare-style rust harness).
- [ ] Root-cause in `encode_with_added_tokens` / SP encoder segment handling; HF semantics: added-token split segments keep their whitespace; SP metaspace converts each segment with `▁` handling — tokie likely drops the whitespace-only `▁` before/after specials.
- [ ] Fix + regression test (accuracy-test style, TinyLlama fixture); run full accuracy suite.

### Task 6: Final verification

- [ ] `cargo test -p pretokie && cargo test -p tokie --features build`
- [ ] `cargo test -p tokie --test accuracy --features build -- --ignored`
- [ ] owt_compare gpt2 + DeepSeek-V3: 20342/20342 both.
- [ ] profile_hotpath before/after table; encode_batch + flat numbers vs gigatoken's 1.9 GB/s reference; update memory file.
