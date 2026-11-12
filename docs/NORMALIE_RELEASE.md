# normalie 0.1.0 release runbook

This runbook covers the first publish of `normalie` only. It does not publish
`pretokie`, `tokie`, Python wheels, or a GitHub release. Nothing here authorizes
a publish on its own.

## Current state

- As checked on 2026-11-11, `GET https://crates.io/api/v1/crates/normalie` returned
  404, so the name appeared unclaimed then. Send a descriptive `User-Agent` when
  checking; one request without one was observed returning 403.
- `.github/workflows/release.yml` publishes `pretokie` and then `tokie` on a
  `release: published` event, and uploads PyPI wheels from the same event. It has
  no `normalie` step. Do not invoke that workflow, and do not bootstrap normalie
  through it.
- Trusted publishing is not configured for `normalie`. A brand-new crate needs a
  first publish with an API token; OIDC trusted publishing cannot be the first
  upload. Reference: <https://crates.io/docs/trusted-publishing/>.

## Prerequisites

- A clean, reviewed, approved `main` commit containing this crate.
- A human operator with a crates.io account whose email is verified. The account
  that first publishes becomes the owner; the operator chooses the owners. Do not
  assume a GitHub token authenticates to crates.io.
- Explicit user authorization for the real publish, after review.

## Preparation (no token, no upload)

Run from the repository root. A clean checkout needs no extra flags. If the
preparation tree is uncommitted, add `--allow-dirty` to `cargo package --list`,
`cargo package`, and the dry run; do not add it to the real publish.

```bash
cargo package --list -p normalie
cargo package -p normalie                 # writes target/package/normalie-0.1.0.crate
cargo +stable test --locked -p normalie
cargo +stable test --doc -p normalie --locked
cargo +stable publish --dry-run -p normalie
```

The dry run needs no crates.io token and does not upload.

The package must contain `README.md`, `src/`, `tests/`, `tests/data/t5.charsmap`,
and both license files. The tests read the charsmap fixture locally, so the
packaged crate is testable on its own.

Extract the actual crate into a fresh temporary directory and test it outside the
repository. This proves the package does not rely on workspaces or paths that only
exist in the checkout. Use a new directory and do not delete it recursively.

```bash
pkg=/tmp/opencode/normalie-package-$(date +%s)
mkdir -p "$pkg"
tar -xzf target/package/normalie-0.1.0.crate -C "$pkg"
cd "$pkg/normalie-0.1.0"
cargo +stable test
```

## Real publish (explicit user authorization only)

Only after commit, review, and a clean approved `main`:

1. Create a short-lived crates.io API token with the `publish-new` scope.
2. Run `cargo login` interactively as the human operator; interactive login is
   preferred for this bootstrap. Keep the token out of chat, logs, source, and
   shell history, and do not pass `--token` as a command-line argument. A CI job
   supplying a short-lived secret through the environment is the normal
   automation path; this manual first publish does not need one.
3. From the approved clean tree, run:

   ```bash
   cargo +stable publish -p normalie
   ```

   Do not pass `--allow-dirty`. Do not publish the workspace.
4. Verify the crate at <https://crates.io/crates/normalie> and the docs build at
   <https://docs.rs/normalie>. docs.rs can take a few minutes after the index
   update.
5. Revoke the bootstrap API token immediately.
6. Configure a trusted publisher on crates.io: GitHub owner `feyninc`, repository
   `tokie`, workflow `release.yml`. No environment is configured today. This
   applies to later versions, not to the first upload.

## Later, separate change

`release.yml` must gain a `normalie` publish step before `pretokie` and `tokie`
in a separate reviewed change, after the first manual publish exists. Do not add
it here, and do not add release, tag, or autopublish code as part of this
preparation.

That step alone does not unblock `tokie`: the future `tokie` `Cargo.toml` path
dependencies must also point at registry versions, `normalie` 0.1.0 and
`pretokie` 0.2.0. The CLI, if published, needs `tokie` 0.2.0. Do not edit those
manifests now.

Do not change any other crate's version for this release.
