# Benchmarks

One OpenWebText sample, 200,041,585 bytes. Release build with fat LTO, one codegen unit, and clang. One warmup, then the median of five passes. The machine has AVX-512 with VBMI2.

pretokie and gigatoken 0.10.0 fill spans in batches of 256. Hugging Face tokenizers v1.0.0-rc.3 writes every span in one `pre_tokenize` call, on its native bitcannon path. The `fancy-regex` feature is off. Where a cell has a rate, that run produced the same pieces as pretokie.

Rates move a few percent between runs. This table is one run.

| cut | pretokie MB/s | gigatoken MB/s | HF tokenizers v1 MB/s | pieces |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | 2029.2 | 2040.0 | 995.8 | 41,621,746 |
| SmolLM | 2049.8 |  |  | 43,288,915 |
| cl100k | 1969.1 | 1639.3 | 908.9 | 39,961,220 |
| Voyage | 2016.7 | 1696.9 | 898.2 | 40,928,464 |
| Olmo | 1927.1 | 1627.0 | 898.9 | 39,961,220 |
| Qwen | 2012.9 | 1700.5 |  | 40,924,497 |
| o200k | 1766.3 | 1504.3 | 565.7 | 39,842,605 |
| Nemotron | 1963.4 | 1598.6 | 584.5 | 41,023,430 |
| Kimi | 1801.2 | 1521.7 | 540.3 | 39,849,233 |
| DeepSeek | 1237.7 | 664.6 | 846.4 | 40,342,055 |
| BERT | 1699.1 |  | 876.7 | 40,611,737 |
| T5 | 1861.1 |  | 1097.1 | 33,032,565 |

An empty cell means that library has no cut for the row.

The gigatoken types are FastR50k, FastCl100k, FastQwen2, FastOlmo3, FastQwen35, FastO200k, FastNemotron, FastKimi, and FastDeepSeekV3. FastDeepSeekV3 is scalar. BERT and T5 have no gigatoken pretokenizer.

SmolLM has no gigatoken cut. FastR50k is the GPT-2 cut: on this file it ran at 2016.3 MB/s and produced 41,621,746 pieces, the GPT-2 count. SmolLM isolates digits, so it produced 43,288,915 pieces.

Hugging Face tokenizers v1 maps the rows it has onto these native grammars:

| cut | HF grammar |
| --- | --- |
| GPT-2 | GPT-2 |
| cl100k, Olmo | cl100k, digits in groups of up to three |
| Voyage | cl100k, one digit at a time |
| o200k | o200k |
| Nemotron | tekken |
| Kimi | Kimi |
| DeepSeek | the three-split sequence, one native pass |
| BERT | `BertPreTokenizer` |
| T5 | `WhitespaceSplit` |

Qwen's letter class includes marks, and that pattern is not one of the HF grammars. SmolLM is not one of them either.
