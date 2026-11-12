# normalie

`normalie` rewrites text into the normal form a tokenizer expects. It is the
normalization crate in the tokie workspace: normalie rewrites the string,
pretokie splits it, and tokie emits ids.

```toml
[dependencies]
normalie = "0.1.0"
```

Each formula is its own type, such as `Nfc`, `Clean`, `BertUncased`, or
`Precompiled`. Formula types are zero-sized except the charsmap holders,
`Precompiled` and `PunctPad`. Text goes in, and a caller-owned `String` receives
the changed form.

## The write contract

Every formula exposes the same call (an associated function on zero-sized
formulas, an instance method on `Precompiled` and `PunctPad`):

`write(text: &str, at_start: bool, out: &mut String) -> bool`

- Pass an empty or cleared `out`. A nonempty `out` is unsupported: it is not a
  result buffer and must not be passed.
- A `false` return means the text was already normalized. `out` is empty, and
  the caller must use the original `text` unchanged.
- A `true` return means the result is `out`. The result may itself be empty
  when the formula deletes all input.
- Never choose between `text` and `out` with `out.is_empty()`. `false` with an
  empty buffer and `true` with an empty buffer mean different things.
- Empty input returns `false` and leaves `out` empty.

`at_start` is true for a whole input and for a segment that begins at byte 0.
Only `MetaspaceFirst` makes its leading `▁` conditional on the flag: with
`false` it adds no prefix (an existing leading space still becomes `▁`).
`Metaspace` ignores the flag and always adds one leading `▁` on a nonempty call
unless the text already starts with a space or `▁`; a leading space becomes that
mark rather than a second prefix. `Nfc`, `Nfd`, and `BertUncased` ignore the
flag too.

## Examples

A BERT formula rewrites and returns `true`:

```rust
let mut out = String::new();
let changed = normalie::BertUncased::write("Hello World", true, &mut out);
assert!(changed);
assert_eq!(out, "hello world");
```

When the text is already in the target form, the return is `false`, the buffer
stays empty, and the caller keeps the original:

```rust
let mut out = String::new();
let text = "caf\u{00E9}";
let changed = normalie::Nfc::write(text, true, &mut out);
assert!(!changed);
assert!(out.is_empty());
let normalized = if changed { out.as_str() } else { text };
assert_eq!(normalized, "caf\u{00E9}");
```

A changed-to-empty result still returns `true`, so do not branch on
`out.is_empty()`:

```rust
let mut out = String::new();
assert!(normalie::Clean::write("\0", true, &mut out));
assert!(out.is_empty());

let mut out = String::new();
assert!(normalie::Strip::write("   ", true, &mut out));
assert!(out.is_empty());
```

Chains run through `Scratch`, which owns two buffers and reports which string
the caller should read. Here NFC leaves the precomposed `é` untouched, then NFD
decomposes it:

```rust
let mut scratch = normalie::Scratch::new();
let out = scratch.apply2(
    "\u{00E9}",
    true,
    normalie::Nfc::write,
    normalie::Nfd::write,
);
assert_eq!(out, "e\u{0301}");
```

Three steps work the same way. NFD decomposition, accent stripping, and
lowercasing turn `CAFÉ` into `cafe`:

```rust
let mut scratch = normalie::Scratch::new();
let out = scratch.apply3(
    "CAF\u{00C9}",
    true,
    normalie::Nfd::write,
    normalie::StripAccents::write,
    normalie::Lower::write,
);
assert_eq!(out, "cafe");
```

`Metaspace` ignores the flag; `MetaspaceFirst` honors it:

```rust
let mut out = String::new();
assert!(normalie::Metaspace::write("hello", false, &mut out));
assert_eq!(out, "▁hello");

let mut out = String::new();
let changed = normalie::MetaspaceFirst::write("hello", false, &mut out);
assert!(!changed);
assert!(out.is_empty());
let kept = if changed { out.as_str() } else { "hello" };
assert_eq!(kept, "hello");
```

## Documentation

- [Architecture](https://github.com/feyninc/tokie/blob/main/crates/normalie/docs/ARCHITECTURE.md)
- [Normalizers and formulas](https://github.com/feyninc/tokie/blob/main/crates/normalie/docs/NORMALIZERS.md)

## Platforms and scope

- x86-64: the scan picks AVX-512, AVX2, or scalar at runtime.
- aarch64: the scan picks NEON or scalar at runtime.
- Scalar paths are always available.
- The crate uses `std`. It is not `no_std`.

This README does not promise performance numbers. The workspace benchmark
notes are in the repository.

## MSRV

`normalie` 0.1.0 does not declare or promise an MSRV. It is built and tested on
current stable Rust (1.98 when this crate was prepared). No `rust-version` field
is set.

## License

MIT OR Apache-2.0. See [LICENSE-MIT](https://github.com/feyninc/tokie/blob/main/crates/normalie/LICENSE-MIT)
and [LICENSE-APACHE](https://github.com/feyninc/tokie/blob/main/crates/normalie/LICENSE-APACHE).
