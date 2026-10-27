# Architecture

Tokie turns text into token ids. Normalie rewrites the string. Pretokie splits that string into spans. Tokie maps each span to ids.

Offsets index the normalized string. The caller owns the id buffer and the scratch.

## Call

```rust
let mut tokenizer = Tokenizer::open("tokenizer.json")?;
let mut ids = Vec::new();
tokenizer.encode("Hello world", &mut ids)?;
```

`encode` appends. It does not clear `ids`. The tokenizer keeps the piece cache and the scratch. `wrap` adds the single-sequence template. `decode` writes text for ids. `batch` encodes many texts on forked tokenizers.

## Steps

1. Added tokens, when the model has them, come out of the raw text first. The gaps are the segments.
2. Each segment is normalized into a caller buffer. A `false` return from normalie means the segment was already in form, and the split reads the original segment.
3. Pretokie writes `Span`s for that segment. A short buffer is resumed with `walk`.
4. Each span is a cache key. A hit appends the stored ids. A miss runs BPE, WordPiece, or Unigram and stores the ids.

Byte-level BPE maps the span through `normalie::ByteLevel` before the merges. The cache key is the span before that map.

## Cache

A piece of 15 bytes or fewer packs into a `u128`: the length in the high byte, the bytes below it. The table is open addressing. A piece longer than 15 bytes is keyed by its bytes in a second table.

The scratch for a miss is reused. The first encode of a piece still builds its symbol list.

## Tree

```text
crates/tokie/src/lib.rs
crates/tokie/src/cut.rs          which pretokie type
crates/tokie/src/norm.rs         which normalie formula
crates/tokie/src/prepare.rs      normalize, then fill spans
crates/tokie/src/cache.rs        piece to ids
crates/tokie/src/bpe.rs          merge loop
crates/tokie/src/wordpiece.rs
crates/tokie/src/unigram.rs
crates/tokie/src/tokenizer.rs  load tokenizer.json and encode
bench/tokie/                     tokenizers v1 oracle and the speed table
```

`open` and `json` read `tokenizer.json`. `pretrained` reads the repo's `.tokie` file from the Hub, whatever its name. A repo with no `.tokie` file uses `tokenizer.json`. Load picks the cut and the normalizer, and builds the BPE pair table, the WordPiece vocab, or the Unigram lattice, plus the id-to-bytes table `decode` reads. Byte-level BPE, WordPiece, Unigram, and unaffixed SentencePiece BPE encode and decode. Affixed SentencePiece BPE is refused.

## Other crates

Normalie normalizes. Pretokie splits. Those crates keep their own oracles. A pretokie change for the cache keeps the spans those oracles lock.
