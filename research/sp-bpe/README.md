# SentencePiece-BPE encoder research harness

Standalone, dependency-free harness used to measure optimization candidates for
`crates/tokie/src/encoder/sentencepiece.rs`. Findings:
[`docs/superpowers/research/2026-07-26-sentencepiece-bpe-efficiency.md`](../../docs/superpowers/research/2026-07-26-sentencepiece-bpe-efficiency.md).

`sp_harness2.rs` ports tokie's `SentencePieceBPE` verbatim as variant A, then
layers one isolated change per variant so each win is attributable. Every
variant is asserted token-identical to A, and — when given a reference id file
— to HF `tokenizers` directly.

It deliberately has **no crate dependencies** so it builds with plain `rustc`
in environments where the crates.io registry is unavailable.

## Running

```sh
pip install tokenizers                      # trainer + correctness reference

# 1. build a corpus (any UTF-8 text; the findings used system documentation)
mkdir -p corpus && cat /usr/share/doc/**/* > corpus/clean.txt
head -c 8388608 corpus/clean.txt > corpus/bench.txt

# 2. train an SP-BPE model (Metaspace + BPE) and export flat vocab/merges
python3 train_hf.py 32000 spbpe_hf32k

# 3. dump HF ground-truth ids
python3 -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('spbpe_hf32k.json')
open('hf_ids.txt','w').write('\n'.join(map(str, t.encode(open('corpus/bench.txt').read()).ids)))"

# 4. run the ladder
rustc --edition 2021 -O -C target-cpu=native -o sp_harness2 sp_harness2.rs
./sp_harness2 spbpe_hf32k.vocab.tsv spbpe_hf32k.merges.tsv corpus/bench.txt 5 hf_ids.txt
```

The harness reports: identity vs A and vs HF, the cold ladder, fixed-cost
diagnostics (split-only and split+lookup floors), document-sized throughput,
a front-table size sweep, and the warm ladder.

`sp_harness.rs` is the earlier three-variant version, kept because it is
smaller and easier to read as an introduction to the structure.
