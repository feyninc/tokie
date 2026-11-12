#!/usr/bin/env python3
"""Validate a tokie release version against the tag and every manifest.

The release workflow passes either a published GitHub release tag (``v0.2.0``)
or a manual version input (``0.2.0``). This script rejects anything that is not
a strict SemVer release before a later job can write to a registry.

It checks:

* the tag is exactly ``v`` plus the version;
* every workspace member has the expected version (normalie is 0.1.0);
* ``crates/tokie-python/pyproject.toml`` has the same version;
* ``tokie`` pins ``normalie`` 0.1.0 and ``pretokie`` to the release version;
* ``tokie-cli`` pins ``tokie`` to the release version.

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
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

NORMALIE_VERSION = "0.1.0"
EXPECTED_PACKAGES = ("normalie", "pretokie", "tokie", "tokie-python", "tokie-cli")


class VersionError(Exception):
    """A version or manifest does not match the release contract."""


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_semver(value: str, label: str) -> str:
    if not isinstance(value, str) or not SEMVER.fullmatch(value):
        fail(f"{label} {value!r} is not a strict SemVer release version")
    return value


def version_from_tag(tag: str) -> str:
    if not tag.startswith("v"):
        fail(f"release tag {tag!r} must start with 'v' (for example v0.2.0)")
    return parse_semver(tag[1:], "release tag")


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


def check_versions(root: Path, version: str) -> None:
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

    for package in EXPECTED_PACKAGES:
        manifest = root / "crates" / package / "Cargo.toml"
        found = package_version(manifest, package)
        expected = NORMALIE_VERSION if package == "normalie" else version
        if found != expected:
            fail(f"{package} version is {found!r}, expected {expected!r}")

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

    tokie_manifest = root / "crates" / "tokie" / "Cargo.toml"
    normalie_requirement = dependency_requirement(tokie_manifest, "normalie")
    if not requirement_pins_exact(normalie_requirement, NORMALIE_VERSION):
        fail(
            "tokie must pin normalie "
            f"{NORMALIE_VERSION!r}, found {normalie_requirement!r}"
        )
    pretokie_requirement = dependency_requirement(tokie_manifest, "pretokie")
    if not requirement_pins_exact(pretokie_requirement, version):
        fail(
            f"tokie must pin pretokie {version!r}, found {pretokie_requirement!r}"
        )

    cli_manifest = root / "crates" / "tokie-cli" / "Cargo.toml"
    tokie_requirement = dependency_requirement(cli_manifest, "tokie")
    if not requirement_pins_exact(tokie_requirement, version):
        fail(
            f"tokie-cli must pin tokie {version!r}, found {tokie_requirement!r}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="release version, for example 0.2.0")
    parser.add_argument("--tag", help="release tag, for example v0.2.0")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the checkout containing this script)",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="append version=<version> to this GitHub Actions output file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.version and args.tag:
        tag_version = version_from_tag(args.tag)
        version = parse_semver(args.version, "--version")
        if tag_version != version:
            fail(f"tag version {tag_version!r} does not match --version {version!r}")
    elif args.tag:
        version = version_from_tag(args.tag)
    elif args.version:
        version = parse_semver(args.version, "--version")
    else:
        fail("pass --version or --tag")

    check_versions(args.root, version)

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"version={version}\n")

    print(f"release version: {version}")
    print("normalie: 0.1.0")
    for package in ("pretokie", "tokie", "tokie-python", "tokie-cli"):
        print(f"{package}: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
