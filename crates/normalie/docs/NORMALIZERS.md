# Normalizers

These are the formulas normalie implements. A model runs one formula, or a chain of them. An unknown normalizer type is refused by `normalie::refuse` and is not executed.

`at_start` is true for the segment at byte 0. Empty input stays empty for every formula.

## NFC

Qwen and ModernBERT.

Canonical composition. ASCII is already NFC. `e` + U+0301 becomes `é`.

## NFD

Canonical decomposition. `é` becomes `e` + U+0301. `BertUncased` and `Albert` decompose as part of their own pass. This type is the standalone form.

## NFKC

Compatibility composition. `ﬁ` (U+FB01) becomes `fi`. `①` becomes `1`.

T5 and XLM-RoBERTa use `Precompiled` with their own blob. The blob's per-character decisions differ from this Unicode form. The XLM-RoBERTa map keeps U+00AD and the C1 controls.

## NFKD

Compatibility decomposition. `ﬁ` becomes `fi`. `é` becomes `e` + U+0301. `Albert` runs this and then drops the marks.

## NMT

SentencePiece's NMT fold, as a standalone step.

These code points are removed: U+0001 through U+0008, U+000B, U+000E through U+001F, U+007F, U+008F, U+009F.

These code points become a space: tab, newline, form feed, carriage return, U+1680, U+200B through U+200F, U+2028, U+2029, `▁`, BOM, and U+FFFD.

`\u{0001}a\tb` becomes `a b`. `▁x` becomes ` x`. A T5 charsmap already includes this fold, so the T5 chain is `Precompiled` then `MetaspaceWords`.

## Lower

Unicode lowercase, one character at a time.

`HELLO` becomes `hello`. `Straße` becomes `straße`. `İ` becomes `i` + U+0307. `中文` stays `中文`.

ASCII `A..=Z` is the byte map: set bit `0x20`. Any other letter goes to the case table. The map alone leaves `É` untouched. `Lower` casefolds it to `é`.

## StripAccents

Drop combining marks. Precomposed letters stay.

`cafe` + U+0301 becomes `cafe`. `café` stays `café`. U+07FD stays, because it was not a mark in Unicode 9. U+1CF2 is removed, because it was a mark in Unicode 9.

`BertUncased` and `Albert` decompose first and then drop marks. That is a different result: `café` becomes `cafe`.

## Clean

BERT `clean_text` with no CJK spacing and no case change.

U+0000, U+FFFD, and control characters other than tab, newline, and carriage return are removed. Tab, newline, carriage return, and Unicode spaces become ` `.

`Hello\nWorld` becomes `Hello World`. `a\u{00A0}b` becomes `a b`. `null\0here` becomes `nullhere`.

## BertCased

`Clean`, and a space on each side of a CJK ideograph.

`Hello\nWorld` becomes `Hello World`. `CAFÉ` stays `CAFÉ`. `a中b` becomes `a 中 b`.

bert-base-cased uses this.

## BertUncased

`Clean`, CJK spacing, NFD, drop combining marks, lowercase. One pass.

`Hello World` becomes `hello world`. `CAFÉ` becomes `cafe`. `a中b` becomes `a 中 b`. `Hello\nWorld` becomes `hello world`.

`\n` becomes ` `. U+007F is removed. An unassigned scalar such as U+0378 is kept.

bert-base-uncased uses this.

## Prepend

Write `▁` in front of a non-empty string. The rest of the text is copied.

`Hello` becomes `▁Hello`. ` world` becomes `▁ world`.

The prefix is `▁`. Any other prefix is refused.

## ReplaceSpace

Replace each ASCII space with `▁`. Tabs, newlines, and other spaces stay.

`a b` becomes `a▁b`. ` a ` becomes `▁a▁`.

Gemma's replace normalizer is this formula. Any other pattern is refused.

## CollapseSpaces

A run of two or more ASCII spaces becomes one space. A single space stays. Tabs and newlines stay.

`a  b` becomes `a b`. `a   b` becomes `a b`. `a\t\tb` stays `a\t\tb`.

This is the `" {2,}"` replace. The replacement is a space. The OpenWebText sample has no such run, so on that file the formula only scans.

## Strip

Trim Unicode whitespace from both ends. A one-sided trim is the same formula with that side off. The result is a borrow of the input whenever the trimmed text is a subslice.

`  a  ` becomes `a`. `\n\thi\t\n` becomes `hi`.

## Metaspace

Spaces become `▁`. A leading `▁` is written when the text does not already start with a space or `▁`. Tabs and newlines stay, so they break the space run and leave a `▁` on each side.

`aa bb cc` becomes `▁aa▁bb▁cc`. ` aa bb` becomes `▁aa▁bb`. `aa\tbb  cc` becomes `▁aa\tbb▁▁cc`. `▁hello` stays `▁hello`.

This is the Hugging Face metaspace pretokenizer with `prepend_scheme=always` and whitespace kept.

## MetaspaceFirst

`Metaspace`, and the leading `▁` is written only when `at_start` is set.

`aa bb cc` at the start becomes `▁aa▁bb▁cc`. The same text after an added token becomes `aa▁bb▁cc`. ` aa bb` is `▁aa▁bb` either way, because the leading space is already the mark.

Mistral-7B-v0.1 uses this.

## MetaspaceNever

Spaces become `▁`. Nothing is prepended.

`aa bb cc` becomes `aa▁bb▁cc`. ` aa bb` becomes `▁aa▁bb`.

## MetaspaceWords

Whitespace of every kind is dropped. Each remaining word gets a leading `▁`, including a word that follows a gap. Edges disappear with the whitespace.

`aa bb cc` becomes `▁aa▁bb▁cc`. `aa\tbb  cc` becomes `▁aa▁bb▁cc`. ` aa bb` becomes `▁aa▁bb`. A string that is only whitespace stays empty.

T5 and XLM-RoBERTa run this after their charsmap. `▁` on an all-whitespace result is the wrong string.

## Llama

Prepend `▁`, then turn spaces into `▁`. The prepend happens even when the text starts with a space. One pass.

`Hello world` becomes `▁Hello▁world`. ` world` becomes `▁▁world`.

Llama 2, TinyLlama, and Phi-3 use this. `Metaspace` turns ` world` into `▁world`.

## ByteLevel

Map each input byte to its GPT-2 unicode character. Printable ASCII stays itself. A space becomes U+0120.

`Hello` stays `Hello`. ` ` becomes `Ġ`.

A byte-level BPE encoder can read the original bytes and skip this string. The formula exists for a caller that wants the mapped text.

## Precompiled

The model's SentencePiece `precompiled_charsmap`. The blob is parsed once. ASCII bytes and the `\r\n` grapheme are cached. A 64-byte window copies a run of printable ASCII when the map leaves those bytes alone, and stops on a control, DEL, or a non-ASCII byte so the trie can take the grapheme.

The T5 map is the blob from `google-t5/t5-small`. `Hello` stays `Hello`. The same map on every BMP scalar matches Hugging Face tokenizers v1.

XLM-RoBERTa, T5, and bge-m3 each carry their own blob in this type. The XLM-RoBERTa map keeps U+00AD and the C1 controls.

## PunctPad

`Precompiled`, then one pass that pads ASCII punctuation, collapses whitespace, strips the edges, and writes metaspace.

An ASCII punctuation character is emitted as `▁`, the character, `▁`, with neighboring whitespace merged into those marks. A whitespace run becomes one `▁`. A string that is only whitespace stays empty. The leading `▁` is skipped when the stripped text already starts with `▁`.

`a.b  c` becomes `▁a▁.▁b▁c`.

minishlab/potion-multilingual-128M uses this.

## Albert

NFKD, drop combining marks, drop controls and U+FFFD, lowercase, collapse whitespace, strip, metaspace. One pass.

`  Héllo   Wörld\n` becomes `▁hello▁world`.

## Chains

Each step is a formula above. The chain swaps two caller-owned buffers.

### T5 and XLM-RoBERTa

`Precompiled`, then `MetaspaceWords`.

Whitespace that the charsmap folds away is dropped. `\u{00A0}` becomes the empty string, not `▁`.

### bge-m3

`Precompiled`, then `CollapseSpaces`, then `Metaspace`.

Only ASCII space runs collapse. A trailing space is kept, and it becomes a `▁`. Tabs are not collapsed. The leading `▁` is skipped when the charsmap output starts with a space or `▁`.

## Refused

A replace pattern other than a single ASCII space, or other than a run of two or more ASCII spaces, is refused. A prepend other than `▁` is refused. A normalizer type with no formula above is refused. The message is `no normalizer for type: …`.
