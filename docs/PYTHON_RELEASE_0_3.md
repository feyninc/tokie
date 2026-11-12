# tokie 0.3.0

Python package release. The Rust crates stay at 0.2.0 (`tokie`, `pretokie`, and
`normalie` 0.1.0); the Rust API and its error strings are unchanged. This
release changes the Python bindings only.

```bash
pip install -U tokie                      # or: pip install "tokie>=0.3.0"
```

## Breaking: `decode` and `decode_bytes` take arrays only

`Tokenizer.decode` and `Tokenizer.decode_bytes` accept only a one-dimensional,
native-endian NumPy `uint32` array that is C-contiguous and aligned. Lists,
tuples, `int64`/`uint64`/other dtypes, rank other than one, non-contiguous or
strided views, and foreign byte order are rejected. There is no implicit cast,
no compatibility facade, and no argument that turns lists back on.

The arrays from `encode`, `encode_plus`, and a slice of `batch`'s flat ids are
accepted, as are empty and read-only arrays, C-contiguous views, and `ndarray`
subclasses. Non-array input, other dtypes, rank other than one, and foreign byte
order raise `TypeError`; a non-empty non-contiguous or unaligned array raises
`ValueError`. Gap and unknown ids, including `2**32 - 1`, still raise
`tokie.TokieError`. `decode` still returns `str` and replaces invalid UTF-8;
`decode_bytes` returns raw bytes; `skip=False` keeps specials in both.

## Performance

The binding copies the input ids into one owned buffer while the GIL is held,
then releases the GIL for the core decode. The old path converted the input per
element; this release bulk-copies. The change is the contract and the copy, not
a new speed claim: the 0.3.0 wheel was not re-timed for this release.

## Example

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("openai-community/gpt2")

ids = tokenizer.encode("Hello world")   # uint32 array, template off
print(tokenizer.decode(ids))            # str
print(tokenizer.decode_bytes(ids))      # bytes
```

## Migrating

From Hugging Face `tokenizers`, `encode(text)` returns an `Encoding` and `.ids`
is a Python `list[int]`. Convert it once, explicitly:

```python
import numpy

hf_ids = hf_tokenizer.encode(text).ids
ids = numpy.asarray(hf_ids, dtype=numpy.uint32)  # new aligned C array
tokenizer.decode(ids)
```

`numpy.asarray(values, dtype=numpy.uint32)` on a Python list allocates a fresh
aligned, C-contiguous array, so it is safe. From an arbitrary existing array,
use `numpy.array(values, dtype=numpy.uint32, copy=True, order="C")`;
`numpy.ascontiguousarray` alone leaves an already C-contiguous but misaligned
buffer unaligned. Do not call `.tolist()` before decode; keep it for printing.

## Compatibility

Python 0.3.0 requires arrays; 0.2.0 also accepted lists and tuples. Historical
0.2.0 notes and benchmarks stay valid for that wheel. See
[RELEASE_NOTE_DECODE_UINT32.md](RELEASE_NOTE_DECODE_UINT32.md).

## Benchmarks

The benchmarks use the pre-version-bump development wheel; the binding
implementation is the same, the artifact and version are recorded as measured,
and there is no retiming for this release. See
[BENCHMARK_UINT32_DECODE.md](BENCHMARK_UINT32_DECODE.md).

## License

MIT OR Apache-2.0.
