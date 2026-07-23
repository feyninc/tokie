#!/usr/bin/env python3
"""Regenerate every tokiers/ repo's tokenizer.tkz as format v13 and upload.

For each repo: build from the repo's own tokenizer.json (via tokie's hub path,
which also captures added/special tokens), save as v13, then VERIFY the
reloaded .tkz against HuggingFace tokenizers on 1MB of enwik8 plus an
added-token probe string — the probe only passes if added tokens survived
inside the file, since from_file never sees tokenizer.json. Upload only
repos that pass. tiktoken-style repos whose json HF can't load are verified
for self-consistency (fresh build vs reloaded .tkz) and marked accordingly.

Usage: python scripts/regen_tokiers_v13.py [--dry-run]
"""

import json
import os
import sys
import tempfile

import tokie
from huggingface_hub import HfApi, hf_hub_download

DRY_RUN = "--dry-run" in sys.argv
ORG = "tokiers"

api = HfApi()

with open("benches/data/enwik8", "rb") as f:
    ENWIK = f.read(1_000_000).decode("utf-8", errors="replace")


def probe_string(added):
    """A string interleaving every added token with normal text."""
    parts = ["The quick brown fox"]
    for content in added[:40]:
        parts.append(content)
        parts.append("jumps over 123 dogs")
    return " ".join(parts)


def verify_repo(repo_id, tkz_path, json_path):
    """Return (status, detail). status: 'pass' | 'self' | 'fail'."""
    fresh = tokie.Tokenizer.from_file(tkz_path)

    added = []
    try:
        data = json.load(open(json_path))
        added = [t["content"] for t in data.get("added_tokens", []) if t.get("content")]
    except Exception:
        pass
    probe = probe_string(added)

    try:
        from tokenizers import Tokenizer as HFTok
        hf = HFTok.from_file(json_path)
        hf.no_truncation()
        hf.no_padding()
        for name, text in [("enwik8", ENWIK), ("probe", probe)]:
            a = hf.encode(text, add_special_tokens=False).ids
            b = list(fresh.encode(text, add_special_tokens=False).ids)
            if a != b:
                j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                return "fail", f"{name} mismatch at token {j} (hf {len(a)} vs tkz {len(b)} tokens)"
        return "pass", f"enwik8+probe exact vs HF ({len(added)} added tokens)"
    except Exception as e:
        # HF can't load this json (e.g. tiktoken-style exports) — self-consistency
        built = tokie.Tokenizer.from_pretrained(repo_id)
        for name, text in [("enwik8", ENWIK), ("probe", probe)]:
            a = list(built.encode(text, add_special_tokens=False).ids)
            b = list(fresh.encode(text, add_special_tokens=False).ids)
            if a != b:
                return "fail", f"self-consistency {name} mismatch"
        return "self", f"self-consistent (HF json load failed: {type(e).__name__})"


def main():
    only = os.environ.get("TOKIERS_ONLY")
    repos = sorted(m.id for m in api.list_models(author=ORG))
    if only:
        want = {f"{ORG}/{n}" for n in only.split(",")}
        repos = [r for r in repos if r in want]
    print(f"{len(repos)} repos; dry_run={DRY_RUN}", flush=True)
    results = {"pass": [], "self": [], "fail": [], "error": []}

    outdir = tempfile.mkdtemp(prefix="tokiers_v13_")
    for repo_id in repos:
        name = repo_id.split("/", 1)[1]
        try:
            json_path = hf_hub_download(repo_id, "tokenizer.json")
            tok = tokie.Tokenizer.from_pretrained(repo_id)
            tkz_path = os.path.join(outdir, f"{name}.tkz")
            tok.save(tkz_path)

            status, detail = verify_repo(repo_id, tkz_path, json_path)
            if status == "fail":
                results["fail"].append((name, detail))
                print(f"FAIL  {name}: {detail}", flush=True)
                continue

            if not DRY_RUN:
                api.upload_file(
                    path_or_fileobj=tkz_path,
                    path_in_repo="tokenizer.tkz",
                    repo_id=repo_id,
                    commit_message="Regenerate tokenizer.tkz as format v13 (self-contained: added/special tokens stored in-file)",
                )
            results[status].append((name, detail))
            print(f"{'OK  ' if status == 'pass' else 'SELF'}  {name}: {detail}", flush=True)
        except Exception as e:
            results["error"].append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}", flush=True)

    print("\n=== summary ===", flush=True)
    for k in ("pass", "self", "fail", "error"):
        print(f"{k}: {len(results[k])}", flush=True)
    for k in ("fail", "error"):
        for name, detail in results[k]:
            print(f"  {k.upper()} {name}: {detail}", flush=True)
    return 1 if results["fail"] or results["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
