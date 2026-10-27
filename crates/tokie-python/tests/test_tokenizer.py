"""Ids from the Python bindings match the tokie CLI, which calls the same Rust API."""

import os
import subprocess
from pathlib import Path

import pytest

import tokie

HUB = Path("/home/bhavnick/.cache/huggingface/hub")
MODELS = {
    "gpt2": HUB
    / "models--openai-community--gpt2/snapshots"
    / "607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
    "bert": HUB
    / "models--google-bert--bert-base-uncased/snapshots"
    / "86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json",
    "t5": HUB / "models--t5-small/snapshots/main/tokenizer.json",
}
GPT2 = Path(os.environ.get("TOKIE_GPT2", MODELS["gpt2"]))
ROOT = Path(__file__).resolve().parents[3]
CLI = [
    "cargo",
    "run",
    "--quiet",
    "--manifest-path",
    str(ROOT / "crates" / "tokie-cli" / "Cargo.toml"),
    "--",
    "ids",
]

SAMPLES = [
    "",
    "Hello world",
    "don't",
    "12345",
    "a12b",
    "Hello, World!",
    "<|endoftext|>",
    "The quick brown fox jumps over the lazy dog.",
    "中文",
    "cafe\u0301",
    "a/b",
    "Hello\nworld",
]


def rust_ids(path: Path, texts: list[str], wrap: bool = False) -> list[list[int]]:
    payload = b"".join(f"{len(text.encode())}\n".encode() + text.encode() + b"\n" for text in texts)
    cmd = CLI + [str(path)]
    if wrap:
        cmd.append("--wrap")
    proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    lines = proc.stdout.decode().splitlines()
    assert len(lines) == len(texts)
    out = []
    for line in lines:
        out.append([] if line == "" else [int(part) for part in line.split()])
    return out


@pytest.fixture(scope="module")
def gpt2():
    if not GPT2.is_file():
        pytest.skip("gpt2 tokenizer.json is not on disk")
    return tokie.Tokenizer.open(str(GPT2))


@pytest.fixture(scope="module", params=["gpt2", "bert", "t5"])
def model(request):
    path = MODELS[request.param]
    if not path.is_file():
        pytest.skip(f"{request.param} tokenizer.json is not on disk")
    return request.param, path, tokie.Tokenizer.open(str(path))


def test_encode_matches_rust(model):
    name, path, tok = model
    got = [tok.encode(text).tolist() for text in SAMPLES]
    assert got == rust_ids(path, SAMPLES, wrap=False), name


def test_wrap_matches_rust(model):
    name, path, tok = model
    got = [tok.wrap(text).tolist() for text in SAMPLES]
    assert got == rust_ids(path, SAMPLES, wrap=True), name


def test_gpt2_wrap_matches_encode(gpt2):
    for text in SAMPLES:
        assert gpt2.wrap(text).tolist() == gpt2.encode(text).tolist()


def test_batch_matches_encode(model):
    _, _, gpt2 = model
    ids, ends = gpt2.batch(SAMPLES)
    flat = ids.tolist()
    ends = ends.tolist()
    cursor = 0
    for text, end in zip(SAMPLES, ends):
        assert flat[cursor:end] == gpt2.encode(text).tolist()
        cursor = end
    assert cursor == len(flat)


def test_empty_batch(gpt2):
    ids, ends = gpt2.batch([])
    assert ids.tolist() == []
    assert ends.tolist() == []


def test_decode_roundtrip(gpt2):
    ids = gpt2.encode("Hello world").tolist()
    assert gpt2.decode(ids) == "Hello world"


def test_token_and_id(gpt2):
    assert gpt2.id("Hello") == 15496
    assert gpt2.token(15496) == "Hello"


def test_save_open_roundtrip(gpt2, tmp_path):
    path = tmp_path / "gpt2.tokie"
    gpt2.save(str(path))
    loaded = tokie.Tokenizer.open(str(path))
    assert loaded.encode("Hello world").tolist() == gpt2.encode("Hello world").tolist()
    assert rust_ids(path, ["Hello world"]) == [loaded.encode("Hello world").tolist()]


def test_missing_file():
    with pytest.raises(tokie.TokieError):
        tokie.Tokenizer.open("/nonexistent/tokenizer.json")
