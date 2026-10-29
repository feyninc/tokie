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

