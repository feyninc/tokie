"""Time Python encode against the tokie CLI on one file.

Both calls encode the whole file once per sample. The CLI is the Rust path.
Python should stay close to it: the binding copies the text in, releases the
GIL, and copies the ids out as one array.
"""

import argparse
import subprocess
import time
from pathlib import Path

import tokie

ROOT = Path(__file__).resolve().parents[2]


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def python_mb(path: str, text: str, reps: int) -> tuple[float, int]:
    model = tokie.Tokenizer.open(path)
    ids = model.encode(text)
    samples = []
    for _ in range(reps):
        # Drop the previous array first so encode can reuse its id buffer.
        del ids
        start = time.perf_counter()
        ids = model.encode(text)
        samples.append(time.perf_counter() - start)
    mb = len(text.encode()) / 1_000_000 / median(samples)
    return mb, int(ids.shape[0])


def rust_mb(path: str, file: str, reps: int) -> str:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "crates" / "tokie-cli" / "Cargo.toml"),
            "--",
            "bench",
            path,
            "--file",
            file,
            "--reps",
            str(reps),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr)
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    text = Path(args.file).read_text()
    py_mb, n_ids = python_mb(args.tokenizer, text, args.reps)
    rust = rust_mb(args.tokenizer, args.file, args.reps)
    print(f"python  {py_mb:.1f} MB/s  {n_ids} ids")
    print(f"rust    {rust}")


if __name__ == "__main__":
    main()
