#!/usr/bin/env python3
"""Validate a tokie release version against the tag and every manifest.

The release workflow passes either a published GitHub release tag
(``v0.2.0`` or ``python-v0.3.0``) or a manual version input (``0.3.0``) plus a
target. This script rejects anything that is not a strict SemVer release before
a later job can write to a registry.

``--target all`` checks a coordinated Rust and Python release: every workspace
member except normalie (0.1.0) carries the release version. ``--target python``
checks a Python-only release: only ``tokie-python`` and ``pyproject.toml``
carry the release version, while the Rust crates keep their own versions and
must keep their dependency pins consistent with those actual versions.

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# SemVer 2.0.0, without build metadata. Cargo rejects duplicate versions that
# differ only by build metadata, so a release version must not carry it.
#
# The core regex is ASCII-only and flat. The optional prerelease suffix is one
# character class; parse_semver validates its dot-separated identifiers one at a
# time. A single alternation-heavy prerelease pattern backtracks exponentially
# on input such as "0.0.0---.---.---...", so this split must stay iterative.
SEMVER_CORE = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z.-]+))?"
)
PRERELEASE_IDENTIFIER = re.compile(r"[0-9A-Za-z-]+")

NORMALIE_VERSION = "0.1.0"
EXPECTED_PACKAGES = ("normalie", "pretokie", "tokie", "tokie-python", "tokie-cli")
PYTHON_PACKAGE = "tokie-python"
RELEASE_TARGETS = ("all", "python")
# (manifest package, dependency) pairs whose version pin must match the
# dependency's actual package version, whatever that version is.
DEPENDENCY_EDGES = (
    ("tokie", "normalie"),
    ("tokie", "pretokie"),
    ("tokie-cli", "tokie"),
)


class VersionError(Exception):
    """A version or manifest does not match the release contract."""


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_semver(value: str, label: str) -> str:
    match = SEMVER_CORE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        fail(f"{label} {value!r} is not a strict SemVer release version")

    prerelease = match.group(4)
    if prerelease is not None:
        for identifier in prerelease.split("."):
            if not PRERELEASE_IDENTIFIER.fullmatch(identifier):
                fail(
                    f"{label} {value!r} has an empty or invalid prerelease "
                    f"identifier {identifier!r}"
                )
            if (
                identifier.isascii()
                and identifier.isdigit()
                and len(identifier) > 1
                and identifier[0] == "0"
            ):
                fail(
                    f"{label} {value!r} has a numeric prerelease identifier "
                    f"with a leading zero: {identifier!r}"
                )
    return value


def version_from_tag(tag: str) -> str:
    if not tag.startswith("v"):
        fail(f"release tag {tag!r} must start with 'v' (for example v0.2.0)")
    return parse_semver(tag[1:], "release tag")


def parse_release_tag(tag: str) -> tuple[str, str]:
    if tag.startswith("python-v"):
        return "python", version_from_tag(tag[len("python-") :])
    if tag.startswith("v"):
        return "all", version_from_tag(tag)
    fail(
        f"release tag {tag!r} must start with 'v' or 'python-v' "
        "(for example v0.2.0 or python-v0.3.0)"
    )


def release_tag(target: str, version: str) -> str:
    prefix = "python-" if target == "python" else ""
    return f"{prefix}v{version}"


def read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        fail(f"missing manifest: {path}")
    except tomllib.TOMLDecodeError as exc:
        fail(f"invalid TOML in {path}: {exc}")


def package_version(manifest: Path, expected_name: str) -> str:
    data = read_toml(manifest)
    package = data.get("package")
    if not isinstance(package, dict):
        fail(f"{manifest} has no [package] table")
    name = package.get("name")
    version = package.get("version")
    if name != expected_name:
        fail(f"{manifest} package name is {name!r}, expected {expected_name!r}")
    if not isinstance(version, str):
        fail(f"{manifest} has no string package.version")
    return version


def dependency_requirement(manifest: Path, dependency: str) -> str:
    data = read_toml(manifest)
    dependencies = data.get("dependencies")
    if not isinstance(dependencies, dict) or dependency not in dependencies:
        fail(f"{manifest} has no [dependencies].{dependency}")
    value = dependencies[dependency]
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        version = value.get("version")
        if not isinstance(version, str):
            fail(
                f"{manifest} dependency {dependency!r} has no registry version; "
                "a path-only dependency cannot be published"
            )
        return version
    fail(f"{manifest} dependency {dependency!r} has an unsupported shape")


def requirement_pins_exact(requirement: str, version: str) -> bool:
    """Accept the common exact/caret/tilde spellings used by this workspace."""
    normalized = requirement.strip()
    return normalized in {
        version,
        f"={version}",
        f"^{version}",
        f"~{version}",
    }


def check_workspace_members(root: Path) -> None:
    workspace = read_toml(root / "Cargo.toml")
    members = workspace.get("workspace", {}).get("members")
    if not isinstance(members, list) or not members:
        fail(f"{root / 'Cargo.toml'} has no workspace.members")

    member_names = {Path(member).name for member in members}
    expected_names = set(EXPECTED_PACKAGES)
    if member_names != expected_names:
        fail(
            "workspace members differ from the release contract: "
            f"found {sorted(member_names)}, expected {sorted(expected_names)}"
        )


def check_pyproject(root: Path, version: str) -> None:
    pyproject = read_toml(root / "crates" / "tokie-python" / "pyproject.toml")
    project = pyproject.get("project")
    if not isinstance(project, dict):
        fail("crates/tokie-python/pyproject.toml has no [project] table")
    if project.get("name") != "tokie":
        fail(f"pyproject project.name is {project.get('name')!r}, expected 'tokie'")
    if project.get("version") != version:
        fail(
            "pyproject project.version is "
            f"{project.get('version')!r}, expected {version!r}"
        )


def check_dependency_pins(root: Path, versions: dict[str, str]) -> None:
    for package, dependency in DEPENDENCY_EDGES:
        manifest = root / "crates" / package / "Cargo.toml"
        requirement = dependency_requirement(manifest, dependency)
        actual = versions[dependency]
        if not requirement_pins_exact(requirement, actual):
            fail(
                f"{package} must pin {dependency} {actual!r}, "
                f"found {requirement!r}"
            )


def check_versions(root: Path, version: str, target: str = "all") -> dict[str, str]:
    if target not in RELEASE_TARGETS:
        fail(f"unknown release target {target!r}")

    check_workspace_members(root)
    versions = {
        package: package_version(root / "crates" / package / "Cargo.toml", package)
        for package in EXPECTED_PACKAGES
    }

    if target == "all":
        for package, found in versions.items():
            expected = NORMALIE_VERSION if package == "normalie" else version
            if found != expected:
                fail(f"{package} version is {found!r}, expected {expected!r}")
    else:
        if versions["normalie"] != NORMALIE_VERSION:
            fail(
                f"normalie version is {versions['normalie']!r}, "
                f"expected {NORMALIE_VERSION!r}"
            )
        if versions[PYTHON_PACKAGE] != version:
            fail(
                f"{PYTHON_PACKAGE} version is {versions[PYTHON_PACKAGE]!r}, "
                f"expected {version!r}"
            )

    check_pyproject(root, version)
    check_dependency_pins(root, versions)
    return versions


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="release version, for example 0.3.0")
    parser.add_argument(
        "--tag",
        help="release tag, for example v0.2.0 or python-v0.3.0",
    )
    parser.add_argument(
        "--target",
        choices=RELEASE_TARGETS,
        help="release target; inferred from --tag when omitted (default: all)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the checkout containing this script)",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="append version, tag, and target to this GitHub Actions output file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    target = args.target
    if args.tag:
        tag_target, tag_version = parse_release_tag(args.tag)
        if target is not None and target != tag_target:
            fail(
                f"release tag {args.tag!r} is a {tag_target!r} tag, "
                f"but --target is {target!r}"
            )
        target = tag_target
        if args.version:
            version = parse_semver(args.version, "--version")
            if version != tag_version:
                fail(
                    f"tag version {tag_version!r} does not match "
                    f"--version {version!r}"
                )
        else:
            version = tag_version
    elif args.version:
        version = parse_semver(args.version, "--version")
        target = target or "all"
    else:
        fail("pass --version or --tag")

    versions = check_versions(args.root, version, target)
    tag = release_tag(target, version)

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"version={version}\n")
            handle.write(f"tag={tag}\n")
            handle.write(f"target={target}\n")

    print(f"release target: {target}")
    print(f"release version: {version}")
    print(f"release tag: {tag}")
    for package in EXPECTED_PACKAGES:
        print(f"{package}: {versions[package]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
