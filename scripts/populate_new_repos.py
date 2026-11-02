#!/usr/bin/env python3
"""Populate newly created tokiers repos with original model files + banner + README."""

from huggingface_hub import HfApi, hf_hub_download
import os
import tempfile

api = HfApi()

# Map: tokiers repo name -> original HF repo
NEW_REPOS = {
    # Sentence Transformers / Embedding models
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "gte-small": "thenlper/gte-small",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-embeddings-v2-base-code": "jinaai/jina-embeddings-v2-base-code",
    "voyage-3-large": "voyageai/voyage-3-large",
    # BERT family
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    # GPT-2 / tiktoken
    "gpt2": "openai-community/gpt2",
    "cl100k": "Xenova/gpt-4",
    "o200k": "Xenova/gpt-4o",
    # LLaMA
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-4-Scout-17B-16E": "meta-llama/Llama-4-Scout-17B-16E",
    "CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
    # Mistral
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Mistral-Nemo-Base-2407": "mistralai/Mistral-Nemo-Base-2407",
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    # Phi
    "phi-2": "microsoft/phi-2",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    # Qwen
    "Qwen2-7B": "Qwen/Qwen2-7B",
    # T5 / XLM-R
    "t5-base": "google-t5/t5-base",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    # New models
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "bge-m3": "BAAI/bge-m3",
    "snowflake-arctic-embed-l-v2.0": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "stablelm-2-1_6b": "stabilityai/stablelm-2-1_6b",
    # Qwen3 / Qwen3.5
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
}

# Files to try downloading from the original repo
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

BANNER_PATH = "assets/tokie-banner.png"
ORG = "tokiers"


def make_readme(repo_name, original_repo):
    return f"""---
tags:
- tokie
library_name: tokie
---

<p align="center">
  <img src="tokie-banner.png" alt="tokie" width="600">
</p>

# {repo_name}

Pre-built [tokie](https://github.com/chonkie-inc/tokie) tokenizer for [{original_repo}](https://huggingface.co/{original_repo}).

## Quick Start (Python)

```bash
pip install tokie
```

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("tokiers/{repo_name}")
ids = tokenizer.encode("Hello, world!")
plus = tokenizer.encode_plus("Hello, world!")
print(plus.tolist())
print(tokenizer.decode(plus.tolist()))
```

## Quick Start (Rust)

```toml
[dependencies]
tokie = "0.2"
```

```rust
use tokie::{{Specials, Tokenizer}};

let mut tokenizer = Tokenizer::pretrained("tokiers/{repo_name}").unwrap();
let mut ids = Vec::new();
tokenizer.encode_plus("Hello, world!", &mut ids).unwrap();
let mut text = String::new();
tokenizer.decode(&ids, &mut text, Specials::Keep).unwrap();
println!("{{text}}");
```

## Files

- `tokenizer.tokie` — tokie binary format
- `tokenizer.json` — original HuggingFace tokenizer

## About tokie

**50x faster tokenization, 10x smaller model files, 100% accurate.**

tokie is a drop-in replacement for HuggingFace tokenizers, built in Rust. See [GitHub](https://github.com/chonkie-inc/tokie) for benchmarks and documentation.

## License

MIT OR Apache-2.0 (tokie library). Original model files retain their original license from [{original_repo}](https://huggingface.co/{original_repo}).
"""


success = 0
failed = []

for repo_name, original_repo in sorted(NEW_REPOS.items()):
    repo_id = f"{ORG}/{repo_name}"
    print(f"\n{'='*60}")
    print(f"  {repo_id}  (from {original_repo})")
    print(f"{'='*60}")

    try:
        # 1. Upload banner
        print(f"  Uploading banner...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=BANNER_PATH,
            path_in_repo="tokie-banner.png",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add tokie banner",
        )
        print("OK")

        # 2. Download and upload tokenizer files from original repo
        for fname in TOKENIZER_FILES:
            print(f"  Fetching {fname}...", end=" ", flush=True)
            try:
                local_path = hf_hub_download(
                    repo_id=original_repo,
                    filename=fname,
                )
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {fname} from {original_repo}",
                )
                print("OK")
            except Exception as e:
                if "404" in str(e) or "EntryNotFound" in str(e) or "does not exist" in str(e).lower():
                    print("not found (skipped)")
                else:
                    print(f"WARN: {e}")

        # 3. Upload README
        print(f"  Uploading README...", end=" ", flush=True)
        readme_content = make_readme(repo_name, original_repo)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            f.flush()
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card README",
            )
        os.unlink(f.name)
        print("OK")

        success += 1

    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append((repo_name, str(e)))

print(f"\n{'='*60}")
print(f"=== Summary ===")
print(f"  Success: {success}/{len(NEW_REPOS)}")
print(f"  Failed:  {len(failed)}/{len(NEW_REPOS)}")
if failed:
    print("\n  Failures:")
    for name, err in failed:
        print(f"    {name} -- {err}")
