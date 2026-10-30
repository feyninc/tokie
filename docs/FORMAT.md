# The .tokie file

`.tokie` is the file a `Tokenizer` saves and opens. It stores the tables encode and decode already use, plus one list of the pieces those tables have already encoded. Load copies the bytes into memory and drops the file. Encode then runs on that memory.

## Summary

A Hugging Face `tokenizer.json` is one JSON document. Opening it builds a tree of owned strings, joins every merge, copies every token into the decode tables, and, for a precompiled normalizer, walks all 65,536 BMP scalars. What remains is a shared set of tables and a piece cache that starts empty.

`.tokie` stores those tables after that work is done. Version 1 is a directory of plain blobs. The header names every blob, each blob is checksummed, and the ranges cover the file with no gaps and no overlap. `Tokenizer::open` reads the file. `Tokenizer::save` writes it. The calls after open are the same calls a JSON load provides. `encode`, `wrap`, `decode`, `batch`, and `fork` do not care which path built the tokenizer.

The piece list is the part that changes encode speed. It holds the most frequent pieces for this cut and this vocabulary, up to 128 MiB. Load inserts them into the one piece cache. A miss still merges and inserts. The file keeps the original 128 MiB list, so a later save does not trade common pieces for the rare ones a single run just saw.

## Calling it

`open` reads the first eight bytes. The magic is `TOKIE`, then `0x89`, then CR, then LF. The byte `0x89` is not valid UTF-8, so a text tool treats the file as binary. CR LF changes if a copy rewrites newlines, and the magic then misses. `TOKI` followed by any other bytes is an old `.tkz` file and fails before the JSON parser. Any other UTF-8 file is `tokenizer.json`. The conventional suffix is `.tokie`. Detection is the magic.

```rust
let mut tokenizer = Tokenizer::open("tokenizer.json")?;
tokenizer.save("tokenizer.tokie")?;

let mut again = Tokenizer::open("tokenizer.tokie")?;
let mut ids = Vec::new();
again.encode("Hello world", &mut ids)?;
```

`save` writes a temporary file in the destination directory, syncs it, and replaces the destination. On Linux that replace is `rename`. A crash before the replace leaves the previous file in place.

`save` writes the piece list the tokenizer was given. A tokenizer opened from JSON has none, so its `save` omits the blob. An encode that ran first does not change the bytes on disk. `fork` shares the tables and copies the loaded piece list. Each tokenizer then updates its own copy. `pretrained` downloads the repo's `.tokie` file, whatever its name. A repo with no `.tokie` file is loaded from `tokenizer.json`.

The public surface is `Tokenizer::open` and `Tokenizer::save`. Wire types stay in a private module. The crate root still exports `Tokenizer`, `Error`, and `Specials`.

## Bytes

The file is little-endian. The loader reads integers with explicit byte order. It does not cast a file pointer to a struct. Version 1 is the only accepted version. Reserved fields are zero. The file is at most 512 MiB, and it holds at most 22 blobs.

```text
offset  size  field
0       8     magic = TOKIE, 0x89, CR, LF
8       4     version = 1
12      4     header_bytes = 128
16      4     entry_bytes = 40
20      4     align = 64
24      4     blob_count
28      4     flags = 0
32      4     directory_crc32c
36      4     header_crc32c
40      8     blob_base
48      8     file_len
56      8     reserved = 0
64      32    source_sha256
96      32    reserved = 0
128     40*N  directory
        pad   zeros to blob_base
blob_base     blobs, each 64-byte aligned
file_len      end, no trailing bytes
```

The 32 bytes at offset 64 are the SHA-256 of the source `tokenizer.json`. `open` does not open that JSON. A checker compares the digest with the file a publisher claims. The ids match because `save` writes the tables `json` built.

CRC-32C is Castagnoli. The check value of the ASCII bytes `123456789` is `0xE3069283`. The header CRC covers the 128 header bytes with bytes 36..40 treated as zero. The directory CRC covers the directory only. Each blob CRC covers that blob and ignores alignment padding.

A directory entry is 40 bytes.

| Offset | Size | Field |
| --- | ---: | --- |
| 0 | 2 | tag |
| 2 | 2 | reserved, zero |
| 4 | 4 | count |
| 8 | 4 | stride |
| 12 | 4 | crc |
| 16 | 8 | absolute offset |
| 24 | 8 | length |
| 32 | 8 | reserved, zero |

A fixed blob has `count * stride == length`. A byte blob (`strtab`, `arena`, `charsmap`, `pieces`) has stride 0, count 1, and a non-zero length. An empty payload is an omitted tag.

After sorting entries by offset, the first offset is `blob_base`, each next offset is the previous end rounded up to 64, the gap bytes are zero, and the last end rounded up to 64 is `file_len`. Overlap, a hole, and trailing bytes are errors.

## Blobs

| Tag | Name | Record |
| --- | --- | --- |
| 1 | meta | one 64-byte record |
| 2 | steps | 8 bytes: kind `u16`, code `u16`, reserved `u32` |
| 3 | literals | four `u32`s: from offset, from length, to offset, to length |
| 4 | strtab | UTF-8 bytes |
| 5 | arena | decode bytes, not necessarily UTF-8 |
| 6 | slots | 32 bytes per id, count equals vocab length |
| 7 | id_keys | string span plus the id `id()` returns |
| 8 | pairs | `left`, `right`, `rank`, `id` as four `u32`s |
| 9 | byte_ids | 256 `u32`s, `u32::MAX` means no symbol |
| 10 | spm_bytes | 256 `u32`s, every byte present |
| 11 | spm_chars | scalar `u32`, id `u32` |
| 12 | nodes | unigram node |
| 13 | edges | byte plus child index |
| 14 | added_raw | id plus a string span |
| 15 | added_norm | same |
| 16 | post_prefix | `u32` ids |
| 17 | post_suffix | `u32` ids |
| 18 | charsmap | framed original precompiled blobs |
| 19 | bmp | 65536 `(start, end)` pairs per charsmap |
| 20 | wp | WordPiece string span plus id |
| 21 | pieces | piece bytes plus the ids |

`meta` holds the model, the cut code, four flags (`prefix_space`, `mark`, `strip_space`, `ignore_merges`), the unk id, WordPiece `max_chars`, vocab length, the WordPiece prefix span, and the unigram unk score as `f64` bits. `ignore_merges` is bit 3 of the flag byte. A set bit means a pretok piece whose byte-level spelling is one vocab token is emitted as that token, and the merge list runs only when that spelling is absent. An old file leaves the bit clear, so every piece still merges. The model byte is 1 for BPE, 2 for WordPiece, 3 for Unigram, and 4 for SentencePiece BPE. Cut codes are fixed numbers in the file module, not the order of the Rust enum. The match that writes them is exhaustive, so a new cut fails to compile until it has a code.

A slot stores the lead span, the next span, the token span, and a present bit plus a special bit. A hole uses `u32::MAX` for the offsets and length 0. `pairs` stores rank and id separately because rank is the merge index.

Always present: `meta`, `slots`, `id_keys`. BPE adds `pairs` and `byte_ids`. WordPiece adds `wp`. Unigram adds `nodes` and `edges`. SentencePiece BPE adds `pairs` and `spm_bytes`. `charsmap` and `bmp` appear together, and only when a step is precompiled. A tag that does not belong to that model is an error. Version 1 allows four precompiled charsmaps. `save` fails when a tokenizer has more. `json` can still encode that tokenizer.

Load checks the caps, the checksums, UTF-8, and the id ranges before it allocates a hash map. It then copies the arena, the slots, the nodes, the edges, and the charsmap trie into the existing structs, and inserts pairs and strings into the hash maps encode already probes. It rebuilds the small added-token matcher and the charsmap ASCII cache. It does not walk the 65,536 BMP scalars. The stored BMP is checked for span bounds and for UTF-8. It calls `Cache::new()`, then inserts the `pieces` blob when that tag is present.

After load returns, the file buffer is dropped. Deleting the file does not change ids. `token()` borrows the tokenizer. `Kind` stays an `FxHashMap` of pairs, a WordPiece string map, or an unigram lattice. `Vocab` stays an owned byte arena plus the string map `id` uses. WordPiece keeps both string maps when an added token overwrites a model string, which is what JSON load does. `merge_ids`, WordPiece, unigram segmentation, and decode stay as they are.

The file leaves out the scratch, the thread pool, the pieces encode learned after load, hash-map buckets, and the JSON text. Added-token `lstrip`, `rstrip`, and `single_word` have no fields. Affixed SentencePiece still fails at JSON load, so it never reaches `save`. An unknown ByteLevel regex stays stored as the cut JSON already chose.

Validation finishes before encode runs. Callers do not see a blob tag.

## Piece cache

Encode spends its time on two things once a piece has been seen. The splitter still walks the text. The cache turns the piece into ids. The merge runs the first time a piece appears. On a warm GPT-2 pass of the OpenWebText sample, that first time is about one percent of the pieces, and the second pass of the same text is about twice as fast as the first.

The `pieces` blob is that warm start, written down. It is optional. Tag 21. The budget is 128 MiB of entry cost. A piece of 15 bytes or fewer costs 32 bytes. A longer piece costs 32 bytes plus its length. The sum of those costs is at most 128 MiB. A larger sum is a refusal. The 512 MiB cap on the whole file stays where it is.

Each record is a `u32` piece length, a `u32` id count, the piece bytes, and that many `u32` ids. Every id is inside the vocab. Load inserts each record into the piece cache encode already probes. The hash is computed on this machine, because one CPU uses CRC-32C and another does not. The file stores the piece and the ids. It does not store bucket indexes.

The bytes of a piece are the exact bytes the splitter emitted, including a leading space. A dictionary word and that same word with a leading space are different keys. The list is built for one cut and one vocabulary. A GPT-2 list loaded onto another vocabulary would return the wrong ids, and the id check refuses an id outside the vocab stored beside it.

The builder splits a corpus with that cut, counts the exact piece bytes, sorts by count, encodes each piece once, and stops when the next piece would pass 128 MiB. For GPT-2 the corpus is FineWeb `sample-10BT`, training shards 000 through 013. Those shards contain 9,255,331,560 pieces and 21,073,257 distinct pieces. The full distinct list would be 674 MiB. The 128 MiB prefix keeps 4,139,437 pieces and covers 99.75% of the pieces in the training shards.

Shard 014 was held out. The table is the median of three runs. Rates are input megabytes per second.

| Piece list | Pieces | FineWeb, 20 MB | FineWeb, 200 MB | OpenWebText tail, 20 MB |
| --- | ---: | ---: | ---: | ---: |
| empty | 0 | 281 | 394 | 294 |
| 128 MiB | 4,139,437 | 767 | 678 | 704 |
| 256 MiB | 8,203,469 | 678 | 682 | 606 |
| 512 MiB | 16,092,235 | 684 | 701 | 604 |
| 674 MiB, every piece | 21,073,257 | 589 | 662 | 596 |

128 MiB is the shipped budget. On the 20 MB held-out slice it is about 2.7 times the empty cache. The larger lists add rare pieces and a larger table, and the probe gets slower. A 1024 MiB list was requested and could not be filled. This sample does not have that many distinct pieces.

Encode keeps using that one cache. A miss merges the piece and inserts it. `save` writes the 128 MiB list again. `fork` copies the loaded list, and each side then learns its own new pieces.

## What load refuses

Load refuses a bad version, an unknown tag, a flag bit it does not know, a torn checksum, an overlapping range, a hole, a trailing byte, a file over 512 MiB, more than 22 blobs, a piece list whose cost exceeds 128 MiB, an id outside the vocab, and a tag that does not belong to the model. An old `.tkz` file fails on the magic, before any JSON parse.

CRC-32C catches a torn write. A deliberate CRC collision can still describe some well-formed tokenizer. The SHA-256 at offset 64 is how a checker ties the file to a named `tokenizer.json`.

## Other containers

The file is a directory of blobs because encode already has a measured hash map for pairs. A mapped perfect hash of those pairs would let encode read the file in place. It would also put the file layout into the merge loop, and the change to a cold merge is unmeasured. Load copies the pairs into the map it already uses, then drops the file.

The previous `.tkz` file stored a vocabulary double-array automaton. Encode does not read one. Added tokens stay a small matcher rebuilt at load, because that set is short. Storing the automaton would not make this encode faster.

An `rkyv`, FlatBuffers, or Cap'n Proto archive of the live tables pays off when encode borrows the file. This load copies into owned maps and drops the file. An archived hash map would be a pointer walk.

Safetensors contributed the coverage rule. Every byte of the file sits in exactly one range. The tensor schema does not fit token spans and pair records, and a Hub weights file must not be readable as a tokenizer.

GGUF contributed typed metadata and aligned blobs. GGUF already stores a tokenizer as token strings plus merge strings, and it can embed the original `tokenizer.json`. That keeps the slow representation. A partial GGUF would invite another tool to disagree about the same bytes.

Pickle executes opcodes on load. A tokenizer file is untrusted, so pickle is out.

Parquet, Arrow, and zip decode pages or inflate archives. This load wants a checksum, a copy, and an insert. Zip also invites an expansion bomb. NumPy `.npy` is one array. A SentencePiece `.model` is a protobuf source, and load would still build the tables. bincode and postcard follow the Rust type. A version bump can misread an old file that has no version a human can see.

## Limits

Version 1 rejects unknown versions, tags, and flag bits. Extending the file means version 2. Old `.tkz` files stay unread.

The piece list is one budget, 128 MiB, for the cut and vocabulary in the same file. Another model needs its own list, built with its own cut. FineWeb 2 is the wrong corpus for the GPT-2 list, because that dataset leaves English out.

The stored BMP is trusted after the span and UTF-8 checks. A hostile BMP normalizes differently. It does not run code.
