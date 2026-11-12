#!/usr/bin/env bash
# Publish the tokie crates in dependency order, safely and idempotently.
#
# Usage:
#   publish_crate.sh probe    --registry crates-io --version 0.2.0
#   publish_crate.sh publish  --registry crates-io --version 0.2.0 [--dry-run]
#   publish_crate.sh validate-workspace --version 0.2.0
#   publish_crate.sh parse-index --version 0.2.0 < index-fixture
#   publish_crate.sh parse-repository < api-fixture
#   publish_crate.sh classify-status --status 200
#   publish_crate.sh classify-core-error < cargo-output
#   publish_crate.sh require-token
#
# The script never accepts a registry URL, a crate name, or a token on the
# command line. It only talks to the public crates.io sparse index and API,
# and it only calls `cargo publish` for pretokie and tokie. normalie 0.1.0 must
# already exist; its first publish is a separate manual bootstrap.

set -euo pipefail

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
dry_run="false"

cmd="${1:-}"
if [ -z "$cmd" ]; then
    usage >&2
    exit 2
fi
shift

registry=""
version=""
status=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --registry)
            registry="${2:-}"
            shift 2
            ;;
        --version)
            version="${2:-}"
            shift 2
            ;;
        --status)
            status="${2:-}"
            shift 2
            ;;
        --repo-root)
            repo_root="${2:-}"
            shift 2
            ;;
        --dry-run)
            dry_run="true"
            shift
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

validate_version_arg() {
    if ! python3 - "$version" <<'PY'
import re
import sys

value = sys.argv[1]
pattern = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
if not re.fullmatch(pattern, value):
    sys.exit(1)
PY
    then
        echo "error: --version must be a strict SemVer release version" >&2
        exit 2
    fi
}

case "$cmd" in
    parse-index|parse-repository|require-token|classify-status|classify-core-error)
        ;;
    validate-workspace)
        validate_version_arg
        ;;
    probe|publish)
        if [ "$registry" != "crates-io" ]; then
            echo "error: --registry crates-io is required" >&2
            exit 2
        fi
        validate_version_arg
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

tmp="$(mktemp "${TMPDIR:-/tmp}/tokie-publish.XXXXXX.py")"
trap 'rm -f "$tmp"' EXIT

cat > "$tmp" <<'PY'
import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

INDEX = "https://index.crates.io"
API = "https://crates.io/api/v1/crates"
USER_AGENT = "tokie-release-automation (https://github.com/feyninc/tokie)"
NORMALIE_VERSION = "0.1.0"
CRATES = ("normalie", "pretokie", "tokie")
EXPECTED_VERSIONS = {
    "normalie": NORMALIE_VERSION,
    "pretokie": None,  # set from --version
    "tokie": None,  # set from --version
}
ALLOWED_NORMALIE_REPOSITORIES = {
    "https://github.com/feyninc/tokie",
    "https://github.com/chonkie-inc/tokie",
}

REPO_ROOT = Path.cwd()
DRY_RUN = False


class ReleaseError(Exception):
    pass


class CargoError(ReleaseError):
    def __init__(self, command, stdout, stderr, returncode):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"cargo {' '.join(command)} failed with exit code {returncode}")


def err(message):
    raise ReleaseError(message)


def read_package_version(path):
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        err(f"missing manifest: {path}")
    except tomllib.TOMLDecodeError as exc:
        err(f"invalid TOML in {path}: {exc}")
    package = data.get("package")
    if not isinstance(package, dict):
        err(f"{path} has no [package] table")
    version = package.get("version")
    if not isinstance(version, str):
        err(f"{path} has no string package.version")
    return version


def validate_workspace(version):
    expected = dict(EXPECTED_VERSIONS)
    expected["pretokie"] = version
    expected["tokie"] = version
    for name, want in expected.items():
        manifest = REPO_ROOT / "crates" / name / "Cargo.toml"
        found = read_package_version(manifest)
        if found != want:
            err(
                f"{name} manifest version is {found!r}, but this command asked "
                f"for {want!r}; refusing to publish the wrong version"
            )
    print(
        "workspace versions ok: "
        + ", ".join(f"{name}={want}" for name, want in expected.items())
    )


def crate_path(name):
    lower = name.lower()
    if len(lower) == 1:
        return f"1/{lower}"
    if len(lower) == 2:
        return f"2/{lower}"
    if len(lower) == 3:
        return f"3/{lower[0]}/{lower}"
    return f"{lower[:2]}/{lower[2:4]}/{lower}"


def classify_http_status(code):
    if code == 200:
        return "ok"
    if code == 404:
        return "missing"
    if code == 403:
        err("HTTP 403 forbidden; refusing to treat it as a missing version")
    if 500 <= code <= 599:
        err(f"HTTP {code} server error; refusing to treat it as a missing version")
    err(f"unexpected HTTP status {code}")


def request(url, accept_404=False):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if exc.code == 404:
            if accept_404:
                return 404, body
            err(f"GET {url} returned HTTP 404")
        classify_http_status(exc.code)
    except urllib.error.URLError as exc:
        err(f"GET {url} failed: {exc.reason}")
    except TimeoutError:
        err(f"GET {url} timed out")


def parse_index(text, version):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            err(f"invalid sparse-index JSON: {exc}")
        if not isinstance(row, dict) or "vers" not in row:
            err("sparse-index row is missing 'vers'")
        if row["vers"] == version:
            if row.get("yanked") is True:
                err(f"exact version {version} is yanked on crates.io; refusing to treat it as present")
            return row
    return None


def index_lookup(name, version, cache_bust=False):
    url = f"{INDEX}/{crate_path(name)}"
    if cache_bust:
        url += f"?t={time.time_ns()}"
    status, body = request(url, accept_404=True)
    if status == 404:
        return None
    return parse_index(body, version)


def normalize_repository(repository):
    normalized = repository.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def parse_repository(text):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        err(f"invalid crates.io API JSON: {exc}")
    crate = payload.get("crate") if isinstance(payload, dict) else None
    repository = crate.get("repository") if isinstance(crate, dict) else None
    if not isinstance(repository, str):
        err("normalie has no repository metadata on crates.io; refusing to trust the name")
    normalized = normalize_repository(repository)
    if normalized not in ALLOWED_NORMALIE_REPOSITORIES:
        err(f"normalie is owned by an unexpected repository: {repository!r}")
    return normalized


def check_normalie_repository():
    # The API is limited to one request per second. The sparse index is not.
    time.sleep(1.0)
    status, body = request(f"{API}/normalie")
    if status != 200:
        err(f"crates.io API returned HTTP {status} for normalie")
    print(f"normalie repository ok: {parse_repository(body)}")


def probe(version):
    validate_workspace(version)
    if index_lookup("normalie", NORMALIE_VERSION) is None:
        err(
            "normalie 0.1.0 is not on crates.io. The first publish must be done "
            "manually with a crates.io API token; OIDC trusted publishing cannot "
            "create a new crate. See docs/NORMALIE_RELEASE.md. Refusing to continue."
        )
    check_normalie_repository()

    states = {"normalie": "present"}
    for name in ("pretokie", "tokie"):
        states[name] = "present" if index_lookup(name, version) is not None else "missing"

    need_oidc = any(state == "missing" for state in states.values())
    for name in CRATES:
        shown = NORMALIE_VERSION if name == "normalie" else version
        print(f"{name} {shown}: {states[name]}")
    print(f"need_oidc={'true' if need_oidc else 'false'}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"need_oidc={'true' if need_oidc else 'false'}\n")
            for name in CRATES:
                handle.write(f"{name}_state={states[name]}\n")
    return states


def require_token():
    token = os.environ.get("CARGO_REGISTRY_TOKEN") or os.environ.get("CRATES_IO_TOKEN")
    if not token:
        err(
            "a missing crate needs a crates.io token; set CARGO_REGISTRY_TOKEN "
            "or CRATES_IO_TOKEN"
        )
    return token


def redact(text, token):
    return text.replace(token, "***") if token else text


def run_cargo(args, token=None):
    env = os.environ.copy()
    if token:
        env["CARGO_REGISTRY_TOKEN"] = token
        env["CRATES_IO_TOKEN"] = token
    else:
        env.pop("CARGO_REGISTRY_TOKEN", None)
        env.pop("CRATES_IO_TOKEN", None)
    print("+ cargo " + " ".join(args))
    proc = subprocess.run(
        ["cargo", *args],
        text=True,
        capture_output=True,
        env=env,
        cwd=REPO_ROOT,
    )
    if proc.stdout:
        print(redact(proc.stdout, token), end="")
    if proc.stderr:
        print(redact(proc.stderr, token), end="", file=sys.stderr)
    return proc


def cargo(args, token=None):
    proc = run_cargo(args, token)
    if proc.returncode != 0:
        raise CargoError(args, proc.stdout, proc.stderr, proc.returncode)


def is_core_resolution_error(text):
    """Only retry the tokie dry run for a lagging sparse-index cache."""
    lower = text.lower()
    names_dependency = "normalie" in lower or "pretokie" in lower
    return names_dependency and (
        "failed to select a version for the requirement" in lower
        or "no matching package named" in lower
    )


def core_dry_run():
    """Retry only a known dependency-resolution failure for tokie's dry run."""
    args = [
        "publish",
        "--dry-run",
        "-p",
        "tokie",
        "--locked",
        "--registry",
        "crates-io",
    ]
    for attempt in range(1, 6):
        proc = run_cargo(args)
        if proc.returncode == 0:
            return
        output = (proc.stdout or "") + (proc.stderr or "")
        if attempt < 5 and is_core_resolution_error(output):
            print(
                f"tokie dry run cannot see normalie/pretokie yet "
                f"(attempt {attempt}/5); retrying in 20s"
            )
            time.sleep(20)
            continue
        raise CargoError(args, proc.stdout, proc.stderr, proc.returncode)


def wait_for(name, version, timeout=180, interval=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if index_lookup(name, version, cache_bust=True) is not None:
            print(f"{name} {version} is visible in the sparse index")
            return
        time.sleep(interval)
    err(f"timed out waiting for {name} {version} in the sparse index")


def publish_one(name, version, token, core=False, dry_run=False):
    if core:
        core_dry_run()
    else:
        cargo(
            [
                "publish",
                "--dry-run",
                "-p",
                name,
                "--locked",
                "--registry",
                "crates-io",
            ]
        )
    if dry_run:
        print(f"{name} {version}: dry run only; not uploading")
        return
    try:
        cargo(
            [
                "publish",
                "-p",
                name,
                "--locked",
                "--registry",
                "crates-io",
            ],
            token,
        )
    except ReleaseError:
        # Another publisher may have won the race between the probe and the
        # upload. Re-probe the exact version before deciding the failure is
        # real. A 403 is only ignored when the exact version is now public.
        if index_lookup(name, version, cache_bust=True) is not None:
            print(
                f"{name} {version} appeared after the publish attempt; "
                "treating it as already published"
            )
            return
        raise
    wait_for(name, version)


def publish(version, dry_run=False):
    states = probe(version)
    token = None

    if states["pretokie"] == "missing":
        token = require_token() if not dry_run else None
        publish_one("pretokie", version, token, dry_run=dry_run)
        if dry_run:
            print("dry run stops before tokie; pretokie is not published")
            return
    else:
        print("pretokie is already published; skipping")

    wait_for("normalie", NORMALIE_VERSION)
    wait_for("pretokie", version)

    if states["tokie"] == "missing":
        token = token or (require_token() if not dry_run else None)
        if index_lookup("normalie", NORMALIE_VERSION) is None:
            err("normalie 0.1.0 is not visible in the sparse index")
        if index_lookup("pretokie", version) is None:
            err(f"pretokie {version} is not visible in the sparse index")
        publish_one("tokie", version, token, core=True, dry_run=dry_run)
    else:
        print("tokie is already published; skipping")

    print("crates.io dry run complete" if dry_run else "crates.io publish complete")


def main():
    global REPO_ROOT, DRY_RUN

    cmd = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else ""
    status = sys.argv[3] if len(sys.argv) > 3 else ""
    repo_root = sys.argv[4] if len(sys.argv) > 4 else ""
    dry_run_arg = sys.argv[5] if len(sys.argv) > 5 else "false"

    if repo_root:
        REPO_ROOT = Path(repo_root)
    if not (REPO_ROOT / "crates" / "normalie" / "Cargo.toml").is_file():
        err(f"--repo-root does not look like the tokie workspace: {REPO_ROOT}")
    DRY_RUN = dry_run_arg == "true"

    if cmd == "validate-workspace":
        validate_workspace(version)
        return
    if cmd == "parse-index":
        if not version:
            err("parse-index needs --version")
        row = parse_index(sys.stdin.read(), version)
        print("present" if row is not None else "missing")
        return
    if cmd == "parse-repository":
        print(parse_repository(sys.stdin.read()))
        return
    if cmd == "classify-status":
        if not status:
            err("classify-status needs --status")
        try:
            code = int(status)
        except ValueError:
            err(f"status is not an integer: {status!r}")
        print(classify_http_status(code))
        return
    if cmd == "classify-core-error":
        print("retry" if is_core_resolution_error(sys.stdin.read()) else "fail")
        return
    if cmd == "require-token":
        require_token()
        print("token present")
        return
    if cmd == "probe":
        probe(version)
        return
    if cmd == "publish":
        publish(version, dry_run=DRY_RUN)
        return
    err(f"unknown command {cmd!r}")


if __name__ == "__main__":
    try:
        main()
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
PY

python3 "$tmp" "$cmd" "$version" "$status" "$repo_root" "$dry_run"
