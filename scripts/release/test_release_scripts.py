#!/usr/bin/env python3
"""Focused unit tests for the release helper scripts.

Run with:

    python3 scripts/release/test_release_scripts.py

No test makes a network write. The index parser, HTTP status classifier, and
guard checks are exercised with fixtures and invalid input.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import check_versions  # noqa: E402
import smoke_wheel  # noqa: E402

PUBLISH = HERE / "publish_crate.sh"


def write_sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 0
            tar.addfile(info)


def run_publish(
    args: list[str],
    stdin: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("CARGO_REGISTRY_TOKEN", None)
    env.pop("CRATES_IO_TOKEN", None)
    env.pop("GITHUB_OUTPUT", None)
    return subprocess.run(
        ["bash", str(PUBLISH), *args],
        input=stdin,
        text=True,
        capture_output=True,
        cwd=cwd or ROOT,
        env=env,
        timeout=60,
    )


def fake_workspace(
    normalie: str = "0.1.0",
    pretokie: str = "0.2.0",
    tokie: str = "0.2.0",
) -> tuple[tempfile.TemporaryDirectory, Path]:
    temp = tempfile.TemporaryDirectory(prefix="tokie-fake-workspace-")
    root = Path(temp.name)
    for name, version in (
        ("normalie", normalie),
        ("pretokie", pretokie),
        ("tokie", tokie),
    ):
        crate = root / "crates" / name
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "{version}"\n',
            encoding="utf-8",
        )
    return temp, root


class VersionTests(unittest.TestCase):
    def test_repo_manifests_match_020(self) -> None:
        check_versions.check_versions(ROOT, "0.2.0")

    def test_wrong_release_version_is_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            check_versions.check_versions(ROOT, "0.3.0")

    def test_semver_rejects_shell_metacharacters(self) -> None:
        for value in ("0.2.0; rm -rf /", "0.2", "v0.2.0", "0.2.0+local"):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                check_versions.parse_semver(value, "version")

    def test_tag_requires_v_prefix(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            check_versions.version_from_tag("0.2.0")
        self.assertEqual(check_versions.version_from_tag("v0.2.0"), "0.2.0")

    def test_semver_prerelease_identifiers(self) -> None:
        for value in ("0.2.0-0", "0.2.0-rc.1", "0.2.0-01a"):
            with self.subTest(value=value):
                self.assertEqual(check_versions.parse_semver(value, "version"), value)
        for value in ("0.2.0-01", "0.2.0-rc.01"):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                check_versions.parse_semver(value, "version")

    def test_semver_long_adversarial_prerelease_is_rejected(self) -> None:
        # Regression: the former alternation-heavy prerelease pattern
        # backtracked exponentially on repeated "--." fragments.
        value = "0.0.0-" + "--." * 3333
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            check_versions.parse_semver(value, "version")

    def test_semver_allows_hyphen_only_prerelease_identifiers(self) -> None:
        value = "0.0.0-0.--.--"
        self.assertEqual(check_versions.parse_semver(value, "version"), value)

    def test_exact_requirement_spellings(self) -> None:
        for requirement in ("0.1.0", "=0.1.0", "^0.1.0", "~0.1.0"):
            self.assertTrue(
                check_versions.requirement_pins_exact(requirement, "0.1.0"),
                requirement,
            )
        for requirement in ("0.1", ">=0.1.0", "0.2.0", "*"):
            self.assertFalse(
                check_versions.requirement_pins_exact(requirement, "0.1.0"),
                requirement,
            )


class WheelTests(unittest.TestCase):
    def test_wheel_version_and_abi(self) -> None:
        info = smoke_wheel.wheel_parts(
            Path("tokie-0.2.0-cp310-cp310-manylinux_2_28_x86_64.whl")
        )
        self.assertEqual(info["version"], "0.2.0")
        self.assertEqual(info["python"], "3.10")
        self.assertEqual(info["abi_tag"], "cp310")

    def test_wheel_with_build_tag(self) -> None:
        info = smoke_wheel.wheel_parts(
            Path("tokie-0.2.0-1-cp314-cp314-pyemscripten_2026_0_wasm32.whl")
        )
        self.assertEqual(info["python"], "3.14")
        self.assertEqual(info["build"], "1")

    def test_wheel_rejects_mismatched_abi(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            smoke_wheel.wheel_parts(
                Path("tokie-0.2.0-cp310-cp314-manylinux_2_28_x86_64.whl")
            )

    def test_wheel_rejects_abi3(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            smoke_wheel.wheel_parts(Path("tokie-0.2.0-py3-none-any.whl"))

    def test_sdist_version(self) -> None:
        self.assertEqual(smoke_wheel.sdist_version(Path("tokie-0.2.0.tar.gz")), "0.2.0")

    SDIST_FILES = (
        "pyproject.toml",
        "Cargo.toml",
        "Cargo.lock",
        "PKG-INFO",
        "README_PYPI.md",
        "src/lib.rs",
        "crates/normalie/Cargo.toml",
        "crates/pretokie/Cargo.toml",
        "crates/tokie/Cargo.toml",
        "crates/tokie-python/Cargo.toml",
    )

    def make_sdist(self, directory: Path, omit: str | None = None) -> Path:
        top = "tokie-0.2.0"
        names = [f"{top}/{name}" for name in self.SDIST_FILES if name != omit]
        archive = directory / "tokie-0.2.0.tar.gz"
        write_sdist(archive, names)
        return archive

    def test_verify_sdist_accepts_full_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tokie-sdist-") as directory:
            archive = self.make_sdist(Path(directory))
            smoke_wheel.verify_sdist(archive, "0.2.0")

    def test_verify_sdist_rejects_missing_tokie_with_pretokie_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tokie-sdist-") as directory:
            archive = self.make_sdist(Path(directory), omit="crates/tokie/Cargo.toml")
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                smoke_wheel.verify_sdist(archive, "0.2.0")
            self.assertIn("tokie workspace crate", stderr.getvalue())

    def test_verify_sdist_rejects_missing_other_crate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tokie-sdist-") as directory:
            archive = self.make_sdist(Path(directory), omit="crates/normalie/Cargo.toml")
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                smoke_wheel.verify_sdist(archive, "0.2.0")
            self.assertIn("normalie workspace crate", stderr.getvalue())

    def test_synthetic_fixture_shape(self) -> None:
        fixture = json.loads(smoke_wheel.FULL_TEXT)
        self.assertEqual(fixture["model"]["type"], "BPE")
        self.assertEqual(fixture["model"]["vocab"]["[PAD]"], 9)
        self.assertEqual(fixture["added_tokens"][0]["id"], 4)
        self.assertEqual(len(fixture["post_processor"]["pair"]), 5)

    def test_truncate_fixture_uses_real_template_ids(self) -> None:
        fixture = json.loads(smoke_wheel.FULL_TEXT)
        specials = fixture["post_processor"]["special_tokens"]
        self.assertEqual(specials["[CLS]"]["ids"], [5])
        self.assertEqual(specials["[SEP]"]["ids"], [6])
        self.assertEqual(fixture["model"]["vocab"]["ab"], 10)


class PublishParserTests(unittest.TestCase):
    def test_parse_index_present(self) -> None:
        proc = run_publish(
            ["parse-index", "--version", "0.1.0"],
            stdin='{"name":"normalie","vers":"0.1.0","yanked":false}\n',
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "present")

    def test_parse_index_missing(self) -> None:
        proc = run_publish(
            ["parse-index", "--version", "0.1.0"],
            stdin='{"name":"normalie","vers":"0.1.1"}\n',
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "missing")

    def test_parse_index_malformed(self) -> None:
        proc = run_publish(
            ["parse-index", "--version", "0.1.0"],
            stdin="not json\n",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("invalid sparse-index JSON", proc.stderr)

    def test_parse_index_yanked_exact_version_fails(self) -> None:
        proc = run_publish(
            ["parse-index", "--version", "0.1.0"],
            stdin='{"name":"normalie","vers":"0.1.0","yanked":true}\n',
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("yanked", proc.stderr)

    def test_classify_core_resolution_error(self) -> None:
        retry_fixtures = (
            'failed to select a version for the requirement `pretokie = "^0.2.0"`',
            'no matching package named `normalie` found',
            'failed to select a version for the requirement `normalie = "^0.1.0"`',
        )
        for fixture in retry_fixtures:
            with self.subTest(fixture=fixture):
                proc = run_publish(["classify-core-error"], stdin=fixture)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout.strip(), "retry")

        fail_fixtures = (
            "error: failed to compile tokie",
            "failed to select a version for the requirement `serde`",
            "HTTP 403 forbidden",
            "no matching package named `serde` found",
        )
        for fixture in fail_fixtures:
            with self.subTest(fixture=fixture):
                proc = run_publish(["classify-core-error"], stdin=fixture)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout.strip(), "fail")

    def test_parse_repository_accepts_expected_owners(self) -> None:
        for repository in (
            "https://github.com/feyninc/tokie",
            "https://github.com/feyninc/tokie.git",
            "https://github.com/chonkie-inc/tokie/",
        ):
            with self.subTest(repository=repository):
                proc = run_publish(
                    ["parse-repository"],
                    stdin=json.dumps({"crate": {"repository": repository}}),
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(
                    proc.stdout.strip(),
                    "https://github.com/chonkie-inc/tokie"
                    if "chonkie-inc" in repository
                    else "https://github.com/feyninc/tokie",
                )

    def test_parse_repository_rejects_unexpected_owner(self) -> None:
        proc = run_publish(
            ["parse-repository"],
            stdin=json.dumps({"crate": {"repository": "https://github.com/evil/tokie"}}),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("unexpected repository", proc.stderr)

    def test_parse_repository_rejects_missing_metadata(self) -> None:
        proc = run_publish(
            ["parse-repository"],
            stdin=json.dumps({"crate": {}}),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no repository metadata", proc.stderr)

    def test_classify_status(self) -> None:
        ok = run_publish(["classify-status", "--status", "200"])
        self.assertEqual(ok.returncode, 0)
        self.assertEqual(ok.stdout.strip(), "ok")

        missing = run_publish(["classify-status", "--status", "404"])
        self.assertEqual(missing.returncode, 0)
        self.assertEqual(missing.stdout.strip(), "missing")

        for code in ("403", "500", "302"):
            with self.subTest(code=code):
                proc = run_publish(["classify-status", "--status", code])
                self.assertEqual(proc.returncode, 1)
                self.assertIn("error:", proc.stderr)


class PublishWorkspaceTests(unittest.TestCase):
    def test_default_repo_root_uses_script_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="outside-repo-") as outside:
            proc = run_publish(
                ["validate-workspace", "--version", "0.2.0"],
                cwd=Path(outside),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("workspace versions ok", proc.stdout)

    def test_matching_fake_workspace_passes(self) -> None:
        temp, root = fake_workspace()
        self.addCleanup(temp.cleanup)
        proc = run_publish(
            ["validate-workspace", "--version", "0.2.0", "--repo-root", str(root)]
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("workspace versions ok", proc.stdout)

    def test_normalie_version_mismatch_fails(self) -> None:
        temp, root = fake_workspace(normalie="0.2.0")
        self.addCleanup(temp.cleanup)
        proc = run_publish(
            ["validate-workspace", "--version", "0.2.0", "--repo-root", str(root)]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("normalie manifest version", proc.stderr)

    def test_pretokie_version_mismatch_fails(self) -> None:
        temp, root = fake_workspace(pretokie="0.1.9")
        self.addCleanup(temp.cleanup)
        proc = run_publish(
            ["validate-workspace", "--version", "0.2.0", "--repo-root", str(root)]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("pretokie manifest version", proc.stderr)

    def test_tokie_version_mismatch_fails(self) -> None:
        temp, root = fake_workspace(tokie="0.1.9")
        self.addCleanup(temp.cleanup)
        proc = run_publish(
            ["validate-workspace", "--version", "0.2.0", "--repo-root", str(root)]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("tokie manifest version", proc.stderr)

    def test_publish_dry_run_aborts_before_cargo_on_mismatch(self) -> None:
        temp, root = fake_workspace(pretokie="0.1.9")
        self.addCleanup(temp.cleanup)
        proc = run_publish(
            [
                "publish",
                "--dry-run",
                "--registry",
                "crates-io",
                "--version",
                "0.2.0",
                "--repo-root",
                str(root),
            ]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("pretokie manifest version", proc.stderr)
        self.assertNotIn("+ cargo", proc.stdout)


class PublishGuardTests(unittest.TestCase):
    def test_require_token_rejects_empty(self) -> None:
        proc = run_publish(["require-token"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("needs a crates.io token", proc.stderr)

    def test_probe_requires_explicit_registry(self) -> None:
        proc = run_publish(["probe", "--version", "0.2.0"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--registry crates-io is required", proc.stderr)

    def test_publish_requires_explicit_registry(self) -> None:
        proc = run_publish(["publish", "--version", "0.2.0"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--registry crates-io is required", proc.stderr)

    def test_invalid_version_is_rejected_before_network(self) -> None:
        proc = run_publish(
            ["probe", "--registry", "crates-io", "--version", "0.2.0; rm -rf /"]
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("strict SemVer", proc.stderr)

    def test_unknown_argument_is_rejected(self) -> None:
        proc = run_publish(
            ["probe", "--registry", "crates-io", "--version", "0.2.0", "--url", "x"]
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown argument", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
