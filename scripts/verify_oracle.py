"""One-shot loop closure: oracle dump vs Python `tokenizers` pre_tokenize_str.

For each scheme: run `examples/dump.rs --impl oracle --text <case>`, convert
HF's ByteLevel-remapped tokens back to bytes, compare exactly.
"""

import struct
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TOKIE = Path.home() / "Workspaces" / "tokie"
DUMP = TOKIE / "target" / "debug" / "examples" / "dump"

REPOS = {
    "r50k": "openai-community/gpt2",
    "cl100k": "Xenova/gpt-4",
    "o200k": "Xenova/gpt-4o",
    "qwen2": "Qwen/Qwen2-7B",
    "voyage": "voyageai/voyage-3",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B",
    "bert": "google-bert/bert-base-uncased",
    "falcon": "tiiuae/falcon-7b",
}
REMAP = {"r50k", "cl100k", "o200k", "qwen2", "voyage", "deepseek", "smollm", "falcon"}

# compact shared edge cases (subset of tests/golden.rs; enough to close the loop)
CASES = [
    "Hello, world!",
    "I'll don't they've we're it's you'd he's I'm won't o'clock",
    "d'Arce O'Brien Nava'i l'Hopital",
    "'s'll 't've",
    " 123 4567 89 0x1F 3.14 1,000,000",
    "a  b   c    d",
    " \n x",
    "a\tb\t\tc",
    " \n\n ",
    "emoji: 😒🌍🎉",
    "日本語テスト 한국어 汉字",
    "café résumé naïve straße",
    "a\u00a0b\u00a0\u00a0x\u2003y\u2028z",
    "<tag attr='val'>text &amp; more</tag>",
    "JSONParser iPhone XMLHttpRequest",
    "path/to/file //usr/bin",
    "1234567 89012345 a1234b",
    "IT'S DON'T O'BILL A'Ll",
    "text...\n\nmore text\n",
    "~abc ~123 ~ ~\nabc",
    "1234567890 a1b2c3",
    "var_name $price #hash @mention",
    "文字混合mixedテスト한국어",
    "café ½ Ⅷ Ⅻ",
    "code: fn main() { let x = 42; } // ctrl",
    "ÉCOLE École",
    "abc123def456 1a2b3c",
    "2024-01-15T00:00:00Z",
    "Hey friend!     How are you?!?",
    "café €100 ±5 → arrow",
    "野口里佳 Noguchi Rika",
    "1234567 890 12 345a",
    "a1b2c3d4",
]


def bytes_to_unicode_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): bytes([b]) for c, b in zip(cs, bs)}


UNI2BYTES = bytes_to_unicode_map()


def bytelevel_to_bytes(token: str) -> bytes:
    return b"".join(UNI2BYTES[ch] for ch in token)


def oracle_dump(scheme: str, text: str) -> list[bytes]:
    res = subprocess.run(
        [str(DUMP), "--scheme", scheme, "--impl", "oracle", "--text", text],
        capture_output=True, check=True,
    )
    buf, pos, toks = res.stdout, 0, []
    (count,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    for _ in range(count):
        (n,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        toks.append(bytes(buf[pos : pos + n]))
        pos += n
    assert pos == len(buf), "trailing dump bytes"
    return toks


def main():
    total_fail = 0
    for scheme, repo in REPOS.items():
        tok_path = hf_hub_download(repo, "tokenizer.json")
        hf_pre = Tokenizer.from_file(tok_path).pre_tokenizer
        fails = 0
        for case in CASES:
            rust_toks = oracle_dump(scheme, case)
            py_toks = [
                bytelevel_to_bytes(t) if scheme in REMAP else t.encode()
                for t, _ in hf_pre.pre_tokenize_str(case)
            ]
            if rust_toks != py_toks:
                fails += 1
                if fails <= 2:
                    print(f"MISMATCH {scheme} {case!r}")
                    print(f"  oracle: {rust_toks[:8]}")
                    print(f"  hf:     {py_toks[:8]}")
        status = "PASS" if fails == 0 else f"{fails}/{len(CASES)} FAIL"
        total_fail += fails
        print(f"{scheme:<9} {status}")
    print()
    print("ORACLE == HUGGINGFACE: all schemes exact" if total_fail == 0 else f"{total_fail} total mismatches — oracle needs fixing")


if __name__ == "__main__":
    main()