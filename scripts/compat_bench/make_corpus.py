"""Build a deterministic OpenWebText sample for the tokie/gigatoken baseline.

Reads the locally cached Skylion007/openwebtext parquet shard. Takes every
STRIDE-th document until TARGET_BYTES of UTF-8 text (full documents only),
so the sample spans the shard instead of taking one contiguous prefix.

Writes:
  <out>/owt_20m.jsonl   one JSON object {"row": <parquet row>, "text": ...}
  <out>/owt_20m.txt     document texts joined with a single "\n"
  <out>/manifest.json   source identity and hashes

The same source bytes are used for every competitor. Run once per baseline.

Usage:
  LD_LIBRARY_PATH=<gcc-lib> PYTHONPATH=<pyarrow-archive> \
    python3 scripts/compat_bench/make_corpus.py --out /tmp/opencode/tokie-compat/corpus
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path(
    "/home/bhavnick/.cache/huggingface/hub/datasets--Skylion007--openwebtext/"
    "blobs/caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e"
)
TARGET_BYTES = 20 * 1024 * 1024
STRIDE = 15


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/opencode/tokie-compat/corpus")
    parser.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(SOURCE)
    table = parquet.read(columns=["text"])
    texts = table.column("text").to_pylist()

    docs = []
    total = 0
    for row in range(0, len(texts), args.stride):
        text = texts[row]
        if text is None:
            continue
        data = text.encode("utf-8")
        docs.append((row, text))
        total += len(data)
        if total >= args.target_bytes:
            break

    jsonl_path = out / "owt_20m.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row, text in docs:
            handle.write(json.dumps({"row": row, "text": text}, ensure_ascii=False))
            handle.write("\n")

    joined = "\n".join(text for _, text in docs)
    joined_bytes = joined.encode("utf-8")
    txt_path = out / "owt_20m.txt"
    txt_path.write_bytes(joined_bytes)

    source_bytes = SOURCE.read_bytes()
    doc_hashes = [sha256_bytes(text.encode("utf-8")) for _, text in docs]
    manifest = {
        "dataset": "Skylion007/openwebtext",
        "parquet": "plain_text/train-00000-of-00080.parquet",
        "source_path": str(SOURCE),
        "source_sha256": sha256_bytes(source_bytes),
        "source_rows": len(texts),
        "selection": {"stride": args.stride, "start": 0, "full_documents": True},
        "documents": len(docs),
        "text_bytes": total,
        "joined_text_bytes": len(joined_bytes),
        "joined_text_sha256": sha256_bytes(joined_bytes),
        "jsonl_bytes": jsonl_path.stat().st_size,
        "jsonl_sha256": sha256_bytes(jsonl_path.read_bytes()),
        "doc_hashes_sha256": sha256_bytes("".join(doc_hashes).encode("ascii")),
        "first_row": docs[0][0],
        "last_row": docs[-1][0],
        "target_bytes": args.target_bytes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
