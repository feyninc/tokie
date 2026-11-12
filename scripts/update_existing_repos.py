#!/usr/bin/env python3
"""Update existing tokiers repos with banner + README (if missing)."""

from huggingface_hub import HfApi
import os
import tempfile

api = HfApi()

# Pre-existing repos and their original HF sources
EXISTING_REPOS = {
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-4-v2": "cross-encoder/ms-marco-MiniLM-L-4-v2",
    "voyage-3": "voyageai/voyage-3",
    "voyage-3-lite": "voyageai/voyage-3-lite",
    "voyage-3.5": "voyageai/voyage-3.5",
    "voyage-3.5-lite": "voyageai/voyage-3.5-lite",
    "voyage-code-2": "voyageai/voyage-code-2",
    "voyage-code-3": "voyageai/voyage-code-3",
    "voyage-finance-2": "voyageai/voyage-finance-2",
    "voyage-law-2": "voyageai/voyage-law-2",
    "voyage-multilingual-2": "voyageai/voyage-multilingual-2",
    "voyage-multimodal-3": "voyageai/voyage-multimodal-3",
    "Cohere-embed-english-v3.0": "Cohere/Cohere-embed-english-v3.0",
    "Cohere-embed-english-light-v3.0": "Cohere/Cohere-embed-english-light-v3.0",
    "Cohere-embed-multilingual-v3.0": "Cohere/Cohere-embed-multilingual-v3.0",
    "Cohere-embed-multilingual-light-v3.0": "Cohere/Cohere-embed-multilingual-light-v3.0",
    "jina-embeddings-v3": "jinaai/jina-embeddings-v3",
    "jina-embeddings-v4": "jinaai/jina-embeddings-v4",
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
    "mxbai-embed-2d-large-v1": "mixedbread-ai/mxbai-embed-2d-large-v1",
    "mxbai-embed-xsmall-v1": "mixedbread-ai/mxbai-embed-xsmall-v1",
    "deepset-mxbai-embed-de-large-v1": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "gte-Qwen2-7B-instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "bge-en-icl": "BAAI/bge-en-icl",
}

BANNER_PATH = "assets/tokie-banner.png"
ORG = "tokiers"


def make_readme(repo_name, original_repo, files):
    binary = ""
    if "tokenizer.tokie" in files:
        binary = "- `tokenizer.tokie` — tokie binary format\n"
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
print(tokenizer.decode(plus, skip=True))
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
tokenizer.decode(&ids, &mut text, Specials::Skip).unwrap();
println!("{{text}}");
```

## Files

{binary}- `tokenizer.json` — original HuggingFace tokenizer (if available)

## About tokie

**50x faster tokenization, 10x smaller model files, 100% accurate.**

tokie is a drop-in replacement for HuggingFace tokenizers, built in Rust. See [GitHub](https://github.com/chonkie-inc/tokie) for benchmarks and documentation.

## License

MIT OR Apache-2.0 (tokie library). Original model files retain their original license from [{original_repo}](https://huggingface.co/{original_repo}).
"""


success = 0
failed = []

for repo_name, original_repo in sorted(EXISTING_REPOS.items()):
    repo_id = f"{ORG}/{repo_name}"
    print(f"  {repo_id:50s} ", end="", flush=True)

    try:
        # Check if README already exists
        try:
            files = [f.rfilename for f in api.list_repo_files(repo_id)]
        except Exception:
            files = []

        # Upload banner if missing
        if "tokie-banner.png" not in files:
            api.upload_file(
                path_or_fileobj=BANNER_PATH,
                path_in_repo="tokie-banner.png",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add tokie banner",
            )

        # Upload/update README
        readme_content = make_readme(repo_name, original_repo, files)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            f.flush()
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update model card README with v0.0.4 API",
            )
        os.unlink(f.name)

        print("OK")
        success += 1

    except Exception as e:
        print(f"FAIL: {e}")
        failed.append((repo_name, str(e)))

print(f"\n=== Summary ===")
print(f"  Success: {success}/{len(EXISTING_REPOS)}")
print(f"  Failed:  {len(failed)}/{len(EXISTING_REPOS)}")
if failed:
    print("\n  Failures:")
    for name, err in failed:
        print(f"    {name} -- {err}")
