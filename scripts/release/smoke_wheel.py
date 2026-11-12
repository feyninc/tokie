#!/usr/bin/env python3
"""Install and smoke-test tokie wheels and source distributions.

Modes
-----

``verify-set``
    Check that a directory holds exactly one wheel for each supported CPython
    (3.10 through 3.14), all at the expected version.

``install-all``
    Find each interpreter with ``uv python find``, create one virtual
    environment per wheel, install the wheel (with its NumPy dependency), and
    run the smoke checks in that environment.

``install``
    Install one wheel or one sdist into a clean virtual environment and run the
    smoke checks. The sdist path proves the source archive builds with Rust.

``check``
    Run the smoke checks in the current interpreter. The Pyodide job calls this
    after installing NumPy and the wheel into a Pyodide virtual environment.

``verify-sdist``
    Check that the source archive contains the workspace crates it needs.

The smoke checks use a synthetic BPE tokenizer. They never download a model or
import pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SUPPORTED_VERSIONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
CP_TAG = re.compile(r"^cp(?P<major>\d)(?P<minor>\d+)(?P<free>t?)$")

# Mirrors tests/test_aliases.py::FULL. A BPE vocab with holes, a raw non-ASCII
# spelling pair, a `[PAD]` spelling, one added special, and both templates.
FULL = {
    "model": {
        "type": "BPE",
        "ignore_merges": True,
        "vocab": {
            "a": 0,
            "b": 1,
            "\u00c3": 2,
            "\u00c3\u00a9": 3,
            "[PAD]": 9,
            "ab": 10,
        },
        "merges": ["a b"],
    },
    "added_tokens": [{"id": 4, "content": "[MASK]", "special": True}],
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [5], "tokens": ["[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [6], "tokens": ["[SEP]"]},
        },
    },
}
FULL_TEXT = json.dumps(FULL)


class SmokeError(Exception):
    """The artifact or environment does not match the release contract."""


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def wheel_parts(path: Path) -> dict[str, str]:
    """Parse a wheel filename from the right, allowing an optional build tag."""
    if path.suffix != ".whl":
        fail(f"not a wheel: {path}")
    parts = path.name[: -len(".whl")].split("-")
    if len(parts) < 5:
        fail(f"unexpected wheel filename: {path.name}")
    name, version = parts[0], parts[1]
    python_tag, abi_tag, platform_tag = parts[-3], parts[-2], parts[-1]
    build_tag = "-".join(parts[2:-3])

    versions = set()
    for tag in python_tag.split("."):
        match = CP_TAG.fullmatch(tag)
        if not match:
            fail(f"unexpected Python tag {tag!r} in {path.name}")
        versions.add(f"{match.group('major')}.{match.group('minor')}")
    if len(versions) != 1:
        fail(f"wheel {path.name} spans several Python versions: {sorted(versions)}")
    python_version = versions.pop()

    if abi_tag != python_tag:
        fail(
            f"wheel {path.name} has ABI tag {abi_tag!r}, expected {python_tag!r}; "
            "refusing to install it into an interpreter it was not built for"
        )

    return {
        "name": name,
        "version": version,
        "python": python_version,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform": platform_tag,
        "build": build_tag,
    }


def sdist_version(path: Path) -> str:
    if not path.name.endswith(".tar.gz"):
        fail(f"not a source distribution: {path}")
    stem = path.name[: -len(".tar.gz")]
    prefix = "tokie-"
    if not stem.startswith(prefix):
        fail(f"unexpected sdist filename: {path.name}")
    return stem[len(prefix) :]


def interpreter_version(python: str) -> str:
    try:
        out = subprocess.check_output(
            [
                python,
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot run interpreter {python!r}: {exc}")
    return out.strip()


def verify_wheel(path: Path, expected_version: str) -> dict[str, str]:
    info = wheel_parts(path)
    if info["version"] != expected_version:
        fail(
            f"wheel {path.name} has version {info['version']!r}, "
            f"expected {expected_version!r}"
        )
    return info


def verify_set(directory: Path, expected_version: str, versions: tuple[str, ...]) -> list[Path]:
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        fail(f"no wheels in {directory}")

    seen: dict[str, Path] = {}
    for wheel in wheels:
        info = verify_wheel(wheel, expected_version)
        python = info["python"]
        if python in seen:
            fail(
                f"two wheels for Python {python}: {seen[python].name} and {wheel.name}"
            )
        seen[python] = wheel

    expected = set(versions)
    found = set(seen)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        fail(f"wheel set mismatch; missing {missing}, unexpected {extra}")

    for python in versions:
        wheel = seen[python]
        info = wheel_parts(wheel)
        print(f"wheel ok: Python {python} {info['platform']} {wheel.name}")
    return [seen[python] for python in versions]


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, env=env)


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def smoke_check(expected_version: str, wheel: Path | None) -> None:
    import importlib.metadata

    import numpy
    import tokie

    if wheel is not None:
        info = verify_wheel(wheel, expected_version)
        running = f"{sys.version_info[0]}.{sys.version_info[1]}"
        if info["python"] != running:
            fail(
                f"wheel {wheel.name} targets Python {info['python']}, "
                f"but this interpreter is {running}"
            )

    installed = importlib.metadata.version("tokie")
    if installed != expected_version:
        fail(f"installed tokie metadata is {installed!r}, expected {expected_version!r}")

    if tokie.__file__ is None:
        fail("tokie has no __file__; it was not imported from an installed artifact")
    module = Path(tokie.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix not in module.parents:
        fail(f"imported tokie from {module}, outside the active environment {prefix}")

    tok = tokie.Tokenizer.json(FULL_TEXT)

    ids = tok.encode("a")
    if ids.dtype != numpy.uint32 or ids.ndim != 1:
        fail(f"encode returned dtype {ids.dtype}, ndim {ids.ndim}")
    if ids.tolist() != [0]:
        fail(f"encode('a') returned {ids.tolist()}, expected [0]")

    batch_ids, ends = tok.batch(["a", "b", "ab", ""])
    if batch_ids.dtype != numpy.uint32 or ends.dtype != numpy.uint64:
        fail(f"batch returned {batch_ids.dtype} and {ends.dtype}")
    if batch_ids.tolist() != [0, 1, 10]:
        fail(f"batch ids are {batch_ids.tolist()}, expected [0, 1, 10]")
    if ends.tolist() != [1, 2, 3, 3]:
        fail(f"batch ends are {ends.tolist()}, expected [1, 2, 3, 3]")

    if tok.decode([3]) != "\u00e9":
        fail("decode([3]) did not return the decoded non-ASCII pair")
    if tok.decode_bytes([2]) != b"\xc3":
        fail("decode_bytes([2]) did not keep the invalid UTF-8 byte")
    if tok.decode_bytes([2, 4, 3]) != b"\xc3[MASK]\xc3\xa9":
        fail("decode_bytes did not keep the special token")
    if tok.decode_bytes([2, 4, 3], skip=True) != b"\xc3\xc3\xa9":
        fail("decode_bytes(skip=True) did not drop the special token")

    vocab = tok.get_vocab()
    expected_vocab = {
        "a": 0,
        "b": 1,
        "\u00c3": 2,
        "\u00c3\u00a9": 3,
        "[PAD]": 9,
        "[MASK]": 4,
        "ab": 10,
    }
    if vocab != expected_vocab:
        fail(f"get_vocab returned {vocab!r}")
    if tok.vocab_size != 11:
        fail(f"vocab_size is {tok.vocab_size}, expected 11")
    if len(vocab) != 7:
        fail(f"get_vocab length is {len(vocab)}, expected 7")
    if tok.pad_token_id != 9:
        fail(f"pad_token_id is {tok.pad_token_id}, expected 9")
    if tok.num_special_tokens_to_add() != 2:
        fail("num_special_tokens_to_add() is not 2")
    if tok.num_special_tokens_to_add(True) != 3:
        fail("num_special_tokens_to_add(True) is not 3")

    if tok.encode_plus("a").tolist() != [5, 0, 6]:
        fail(f"encode_plus('a') is {tok.encode_plus('a').tolist()}")
    if tok.encode_plus("a", pair="b").tolist() != [5, 0, 6, 1, 6]:
        fail(f"encode_plus pair is {tok.encode_plus('a', pair='b').tolist()}")
    if tok.encode_plus("a", pad=True, max_len=5).tolist() != [5, 0, 6, 9, 9]:
        fail(f"encode_plus pad is {tok.encode_plus('a', pad=True, max_len=5).tolist()}")

    # "ab" encodes to one body id [10], so encode_plus is [5, 10, 6]. max_len=2
    # must remove the body entirely, leaving only [CLS] and [SEP].
    full_short = tok.encode_plus("ab").tolist()
    if full_short != [5, 10, 6]:
        fail(f"encode_plus('ab') is {full_short}, expected [5, 10, 6]")
    truncated_short = tok.encode_plus("ab", truncate=True, max_len=2).tolist()
    if len(full_short) <= 2:
        fail("truncate fixture is not longer than max_len=2")
    if truncated_short != [5, 6] or len(truncated_short) != 2:
        fail(f"encode_plus truncate max_len=2 is {truncated_short}, expected [5, 6]")

    # "abab" encodes to [10, 10], so encode_plus is [5, 10, 10, 6]. max_len=3
    # must cut one body id, not merely return the untruncated sequence.
    full_long = tok.encode_plus("abab").tolist()
    if full_long != [5, 10, 10, 6]:
        fail(f"encode_plus('abab') is {full_long}, expected [5, 10, 10, 6]")
    truncated_long = tok.encode_plus("abab", truncate=True, max_len=3).tolist()
    if len(full_long) <= 3:
        fail("truncate fixture is not longer than max_len=3")
    if truncated_long != [5, 10, 6] or len(truncated_long) != 3:
        fail(f"encode_plus truncate max_len=3 is {truncated_long}, expected [5, 10, 6]")

    if tok.token(3) != "\u00c3\u00a9":
        fail(f"token(3) is {tok.token(3)!r}")
    if tok.id("ab") != 10:
        fail(f"id('ab') is {tok.id('ab')!r}")
    if tok.count_tokens("ab") != 1:
        fail(f"count_tokens('ab') is {tok.count_tokens('ab')}")
    if tok.count_tokens_batch(["a", "ab"]) != [1, 1]:
        fail(f"count_tokens_batch is {tok.count_tokens_batch(['a', 'ab'])}")
    if repr(tok) != "Tokenizer()":
        fail(f"repr(tok) is {repr(tok)!r}")
    if hasattr(tokie, "Encoding"):
        fail("the removed Encoding namespace is still exported")

    with tempfile.TemporaryDirectory(prefix="tokie-smoke-") as directory:
        saved = Path(directory) / "tokenizer.tokie"
        tok.save(str(saved))
        reopened = tokie.Tokenizer.open(str(saved))
        if reopened.encode("ab").tolist() != tok.encode("ab").tolist():
            fail("save/open did not round-trip the tokenizer")
        if reopened.vocab_size != tok.vocab_size:
            fail("save/open changed vocab_size")
        # Saving over the same path reruns the temporary write and flush. On
        # Windows the flush needs a writable handle, so a read-only reopen
        # fails here.
        tok.save(str(saved))
        again = tokie.Tokenizer.open(str(saved))
        if again.get_vocab() != tok.get_vocab():
            fail("saving twice changed the tokenizer vocab")
        if again.vocab_size != tok.vocab_size:
            fail("saving twice changed vocab_size")
        leftovers = sorted(path.name for path in Path(directory).iterdir() if path.suffix == ".tmp")
        if leftovers:
            fail(f"save left temporary files behind: {leftovers}")

    print(
        f"smoke ok: tokie {installed} on Python "
        f"{sys.version_info[0]}.{sys.version_info[1]} ({sys.platform})"
    )


def install_one(
    artifact: Path,
    python: str,
    expected_version: str,
    wheel: bool,
) -> None:
    running = interpreter_version(python)
    if wheel:
        info = verify_wheel(artifact, expected_version)
        if info["python"] != running:
            fail(
                f"wheel {artifact.name} targets Python {info['python']}, "
                f"but {python} is Python {running}"
            )
    else:
        found = sdist_version(artifact)
        if found != expected_version:
            fail(
                f"sdist {artifact.name} has version {found!r}, "
                f"expected {expected_version!r}"
            )

    venv = Path(tempfile.mkdtemp(prefix="tokie-venv-"))
    try:
        run_checked([python, "-m", "venv", str(venv)])
        venv_py = venv_python(venv)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_NO_INPUT"] = "1"

        # Install the artifact by path. NumPy is a real project dependency, so
        # pip resolves it from the index unless it is already installed.
        run_checked(
            [str(venv_py), "-m", "pip", "install", "--no-input", str(artifact)],
            env=env,
        )

        check = [str(venv_py), str(Path(__file__).resolve()), "check", "--expected-version", expected_version]
        if wheel:
            check.extend(["--wheel", str(artifact)])
        run_checked(check, env=env)
    finally:
        shutil.rmtree(venv, ignore_errors=True)


def uv_find(version: str) -> str:
    uv = shutil.which("uv")
    if uv is None:
        fail("uv is not on PATH; install it before running install-all")
    try:
        out = subprocess.check_output(
            [uv, "python", "find", version],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"uv python find {version} failed: {exc}")
    path = out.strip()
    if not path:
        fail(f"uv python find {version} returned an empty path")
    return path


def install_all(directory: Path, expected_version: str) -> None:
    wheels = verify_set(directory, expected_version, SUPPORTED_VERSIONS)
    for wheel in wheels:
        info = wheel_parts(wheel)
        python = uv_find(info["python"])
        print(f"installing {wheel.name} with {python}")
        install_one(wheel, python, expected_version, wheel=True)


def verify_sdist(archive: Path, expected_version: str) -> None:
    found = sdist_version(archive)
    if found != expected_version:
        fail(f"sdist {archive.name} has version {found!r}, expected {expected_version!r}")

    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
    except (OSError, tarfile.TarError) as exc:
        fail(f"cannot read {archive}: {exc}")

    top = f"tokie-{expected_version}"
    if not any(name == top or name.startswith(top + "/") for name in names):
        fail(f"sdist does not contain the top-level directory {top}/")

    required = {
        "pyproject.toml": f"{top}/pyproject.toml",
        "Cargo.toml": f"{top}/Cargo.toml",
        "Cargo.lock": None,
        "PKG-INFO": f"{top}/PKG-INFO",
        "README_PYPI.md": f"{top}/README_PYPI.md",
    }
    for label, exact in required.items():
        if exact is not None:
            if exact not in names:
                fail(f"sdist is missing {label} ({exact})")
        elif not any(name.endswith("/Cargo.lock") or name == "Cargo.lock" for name in names):
            fail(f"sdist is missing {label}")

    # Maturin may put the Python crate at the archive root or keep the
    # workspace layout under crates/tokie-python. Accept either.
    if not any(
        name == f"{top}/src/lib.rs"
        or name.endswith("/tokie-python/src/lib.rs")
        for name in names
    ):
        fail("sdist is missing the tokie-python src/lib.rs")

    for crate in ("normalie", "pretokie", "tokie", "tokie-python"):
        manifest = f"{top}/crates/{crate}/Cargo.toml"
        if manifest not in names:
            fail(f"sdist does not bundle the {crate} workspace crate ({manifest})")

    if any(name.endswith(".whl") for name in names):
        fail("sdist unexpectedly contains a wheel")

    print(f"sdist ok: {archive.name} bundles normalie, pretokie, tokie, and tokie-python")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    verify = sub.add_parser("verify-set", help="check a directory of wheels")
    verify.add_argument("--dir", type=Path, required=True)
    verify.add_argument("--expected-version", required=True)
    verify.add_argument(
        "--versions",
        default=",".join(SUPPORTED_VERSIONS),
        help="comma-separated Python versions (default: 3.10..3.14)",
    )

    install_all_parser = sub.add_parser("install-all", help="install and smoke every wheel")
    install_all_parser.add_argument("--dir", type=Path, required=True)
    install_all_parser.add_argument("--expected-version", required=True)

    install = sub.add_parser("install", help="install one artifact in a clean venv")
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", type=Path)
    source.add_argument("--source", type=Path)
    install.add_argument("--python", required=True)
    install.add_argument("--expected-version", required=True)

    check = sub.add_parser("check", help="run the smoke checks in this interpreter")
    check.add_argument("--expected-version", required=True)
    check.add_argument("--wheel", type=Path)

    sdist = sub.add_parser("verify-sdist", help="check a source distribution")
    sdist.add_argument("--archive", type=Path, required=True)
    sdist.add_argument("--expected-version", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.mode == "verify-set":
        versions = tuple(part.strip() for part in args.versions.split(",") if part.strip())
        verify_set(args.dir, args.expected_version, versions)
    elif args.mode == "install-all":
        install_all(args.dir, args.expected_version)
    elif args.mode == "install":
        if args.wheel is not None:
            install_one(args.wheel, args.python, args.expected_version, wheel=True)
        else:
            install_one(args.source, args.python, args.expected_version, wheel=False)
    elif args.mode == "check":
        smoke_check(args.expected_version, args.wheel)
    elif args.mode == "verify-sdist":
        verify_sdist(args.archive, args.expected_version)
    else:
        fail(f"unknown mode {args.mode!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
