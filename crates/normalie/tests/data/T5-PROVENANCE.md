# t5.charsmap provenance

`t5.charsmap` is the base64-decoded value of the `normalizer.precompiled_charsmap`
field in the T5 tokenizer definition. The fixture is the upstream bytes with no
modifications.

## Upstream source

| Item | Value |
| --- | --- |
| Model | [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small) |
| Revision | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| Pinned file URL | https://huggingface.co/google-t5/t5-small/resolve/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/tokenizer.json |
| Source field | `normalizer.precompiled_charsmap` (`normalizer.type == "Precompiled"`) |
| Upstream `tokenizer.json` SHA-256 | `d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066` |
| Model card license | Apache-2.0 |

The committed copy at `bench/normalize/data/t5.tokenizer.json` has the SHA-256
above and decodes to this fixture.

## Fixture integrity

| Item | Value |
| --- | --- |
| Size | 237539 bytes |
| SHA-256 | `0942789e0111452f1cc446f70036e26d24458c3c7858c53cc12ca4d87531279b` |

## Reproduction

From the repository root, with the pinned `tokenizer.json` saved or the committed
copy in place:

```bash
python3 - <<'PY'
import base64, hashlib, json

with open("bench/normalize/data/t5.tokenizer.json") as f:
    tokenizer = json.load(f)

data = base64.b64decode(tokenizer["normalizer"]["precompiled_charsmap"])
with open("/tmp/t5.charsmap", "wb") as f:
    f.write(data)

print(len(data))
print(hashlib.sha256(data).hexdigest())
PY
```

Expected output:

```text
237539
0942789e0111452f1cc446f70036e26d24458c3c7858c53cc12ca4d87531279b
```

The same bytes are obtained with `jq` and `base64`:

```bash
jq -r '.normalizer.precompiled_charsmap' bench/normalize/data/t5.tokenizer.json \
  | tr -d '\n' \
  | base64 -d > /tmp/t5.charsmap
sha256sum /tmp/t5.charsmap
```

Confirm that the result matches the committed fixture:

```bash
cmp /tmp/t5.charsmap crates/normalie/tests/data/t5.charsmap
```

No conversion, re-encoding, trimming, or other transformation is applied to the
decoded bytes.

## Licensing

The T5 model repository and its tokenizer data are distributed under
Apache-2.0. The `t5.charsmap` fixture is redistributed under Apache-2.0; it is
not MIT-licensed and must not be described as MIT.

The `normalie` crate's own source code remains `MIT OR Apache-2.0`. Copies of
both license texts are at `crates/normalie/LICENSE-MIT` and
`crates/normalie/LICENSE-APACHE`, matching the repository root byte for byte.

No NOTICE file is included: the upstream model repository at the pinned revision
does not contain one, and none is fabricated here.
