#!/usr/bin/env bash
# References and tokenizer files for bench/tokie.
#
# The bench Cargo.toml paths are siblings of this repo: ../gigatoken and
# ../tokenizers-v1. From bench/tokie that is ../../../gigatoken.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ ! -f ../gigatoken/Cargo.toml ]]; then
  git clone https://github.com/marcelroed/gigatoken.git ../gigatoken
  git -C ../gigatoken fetch --depth 1 origin fac0114b37120ec8a76362e9ee8e1c742aaafaef
  git -C ../gigatoken checkout --detach FETCH_HEAD
fi

if [[ ! -f ../tokenizers-v1/tokenizers/tk-encode/Cargo.toml ]]; then
  git clone https://github.com/huggingface/tokenizers.git ../tokenizers-v1
  git -C ../tokenizers-v1 fetch --depth 1 origin f8f41813f4bf8f3e2a467e31637c8837373b8dfa
  git -C ../tokenizers-v1 checkout --detach FETCH_HEAD
fi

data="bench/tokie/data"
mkdir -p "$data"

fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "$data/$out" ]]; then
    return
  fi
  curl -fL --retry 3 --retry-delay 2 -o "$data/$out.partial" "$url"
  mv "$data/$out.partial" "$data/$out"
}

# Snapshot urls match the files the id check was written against.
fetch "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json" gpt2.json
fetch "https://huggingface.co/Xenova/gpt-4/resolve/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json" cl100k.json
fetch "https://huggingface.co/Xenova/gpt-4o/resolve/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json" o200k.json
fetch "https://huggingface.co/Qwen/Qwen2-7B/resolve/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json" qwen2.json
fetch "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json" qwen.json
fetch "https://huggingface.co/allenai/OLMo-2-1124-7B/resolve/main/tokenizer.json" olmo.json
fetch "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/resolve/main/tokenizer.json" nemotron.json
fetch "https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/tokenizer.json" kimi.json
fetch "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/effd688a12921b4cc83e3312b6feb579f70f9c71/tokenizer.json" smollm.json
fetch "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json" deepseek.json
fetch "https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json" bert-uncased.json
fetch "https://huggingface.co/t5-small/resolve/main/tokenizer.json" t5.json
fetch "https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json" albert.json
fetch "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json" llama.json
