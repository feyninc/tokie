# Patterns

These are the cuts pretokie implements. The text is already normalized. An unknown split pattern is refused by `pretokie::refuse` and is not executed.

## GPT-2

GPT-2, r50k, and p50k.

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Contractions are the lowercase forms `'s`, `'t`, `'m`, `'d`, `'ll`, `'ve`, and `'re`. A space stays on the piece it prefixes. A run of digits stays one piece.

`Hello world` is `Hello` and ` world`. `don't` is `don` and `'t`.

## SmolLM

SmolLM. Each numeric character is its own piece. The GPT-2 pattern then runs on every remaining stretch.

`a12b` is `a`, `1`, `2`, `b`. `don't` is still `don` and `'t`.

## cl100k

cl100k_base, Llama 3, and Olmo. Olmo is this same pattern under its own type.

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Contractions are their own pieces and match in either case. Digits are grouped by up to three. A newline run is its own piece, and a punctuation run may absorb the newlines that follow it.

`12345` is `123` and `45`.

## Voyage

Voyage and Qwen2. The cl100k pattern with one numeric character per piece.

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`12345` is `1`, `2`, `3`, `4`, `5`.

## Qwen

Qwen3.5. Voyage, plus marks continuing a letter, and marks excluded from the punctuation class.

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

## o200k

o200k_base, GPT-4o, Llama 4, and gpt-oss.

```text
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

An uppercase letter starts a piece when the previous letter is lowercase, so `camelCase` is `camel` and `Case`. `HTTPResponse` stays one piece. Contractions are suffixes: `don't` is one piece, and `can'ts` is `can't` and `s`. Digits are grouped by up to three. A punctuation run absorbs the newlines and slashes that follow it, so `.\n//` is one piece.

## Nemotron

Nemotron. Hugging Face tokenizers v1 knows this pattern as tekken.

```text
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The case split is the o200k split. Contractions are ordinary punctuation, so `don't` is `don` and `'t`. Each digit is its own piece. Slashes still absorb into the punctuation run: `.\n//` is one piece.

## Kimi

Kimi K2.

```text
[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Han letters form their own run, so `abc中文def` is `abc`, `中文`, `def`. Contractions are suffixes, and digits are grouped by up to three. A slash stays its own punctuation piece: `.\n//` is `.\n` and `//`.

## DeepSeek

DeepSeek V3. Three isolated splits, applied in order. A stretch that matches none of them survives.

```text
\p{N}{1,3}
```

```text
[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+
```

```text
[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The CJK range is the unified ideographs through U+9FA5, then hiragana and katakana. `!hello` is one piece. `!!hello` is `!!` and `hello`. `don't` is `don` and `'t`. `1234` is `123` and `4`. A whitespace run that ends on a digit or on a CJK character stays whole, so `a  1` is `a`, `  `, `1`.

## BERT

BERT, DistilBERT, ELECTRA, MiniLM, GTE, and BGE. Cased and uncased differ in the normalizer, which has already run.

Whitespace is dropped. Each punctuation character is its own piece. Every other character, including CJK, stays in the run.

```text
[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]|\p{P}|[^\s!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~\p{P}]+
```

`Hello, world!` is `Hello`, `,`, `world`, `!`. `don't` is `don`, `'`, `t`. `18th` is one piece. `中文` is one piece, and `中 文` is `中` and `文`.

## T5

T5, ALBERT, and XLM-RoBERTa, for the whitespace split that runs beside SentencePiece metaspace. The metaspace character is applied by the normalizer.

A piece is a maximal run of non-whitespace.

```text
\S+
```

`Hello, world!` is `Hello,` and `world!`. `don't`, `a-b`, and `$5` each stay one piece.
