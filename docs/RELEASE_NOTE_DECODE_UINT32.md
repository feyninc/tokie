# Decode input: native `uint32` NumPy arrays only

**Status:** Python 0.3.0. This is the contract for the `tokie` Python package 0.3.0 release, which wraps the Rust `tokie` 0.2.0 crate; the Rust crates stay at 0.2.0. Python 0.2.0 accepted lists, so this is a breaking change for Python callers.

Python 0.3.0 `Tokenizer.decode` and `Tokenizer.decode_bytes` accept only a one-dimensional, native-endian NumPy `uint32` array that is C-contiguous and aligned. There is no implicit cast, list conversion, or fallback, and no argument re-enables lists. Pass the array returned by `encode`, `encode_plus`, or a slice of `batch`'s flat ids directly.

The binding copies the ids into an owned `Vec` while the GIL is held, then drops the array borrow and releases the GIL for the core decode. No borrowed NumPy slice crosses the detached call.

## Accepted

- A 1-D native `uint32` NumPy array.
- The arrays returned by `encode` and `encode_plus`.
- A slice of `batch`'s flat `uint32` ids, for example `flat[start:end]`.
- Empty arrays (they decode to `""` / `b""`).
- Read-only arrays.
- C-contiguous views, including ordinary contiguous slices.
- NumPy `ndarray` subclasses.

A subclass is accepted, but only its raw underlying ids are read. Mask values are ignored: a `numpy.ma.MaskedArray` decodes its underlying data, so fill masked ids explicitly before decoding. No masked-array handling is added.

## Rejected

- `TypeError` for a list, tuple, `bytes`, `memoryview`, scalar, or any other non-array input.
- `TypeError` for another dtype: `int64`, `uint64`, `float`, `bool`, and so on.
- `TypeError` for rank other than one, including 0-D and 2-D arrays.
- `TypeError` for foreign byte order; the dtype must be native-endian.
- `ValueError` for a non-empty non-contiguous array, such as a strided or negative-stride view.
- `ValueError` for an unaligned non-empty array.

Gap ids, unknown ids, and `2**32 - 1` still raise `tokie.TokieError`. The decode special-token and invalid-UTF-8 semantics are unchanged: `decode` is lossy for invalid UTF-8, `decode_bytes` keeps raw bytes, and `skip=True` omits specials.

## Migration

Do not call `.tolist()` before decode. `.tolist()` remains fine for printing and assertions.

```python
import numpy
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("openai-community/gpt2")

ids = tokenizer.encode("Hello world")       # uint32 array
print(tokenizer.decode(ids))
print(tokenizer.decode_bytes(ids, skip=True))

flat, ends = tokenizer.batch(["Hello", "world"])
print(tokenizer.decode(flat[: ends[0]]))    # first document

# Ids built by hand need an explicit array. Do not pass the list itself.
manual = numpy.asarray([0, 1], dtype=numpy.uint32)
print(tokenizer.decode(manual))

# An arbitrary existing array: copy to an aligned, C-contiguous buffer.
copied = numpy.array(manual, dtype=numpy.uint32, copy=True, order="C")
print(tokenizer.decode(copied))
```

`numpy.asarray(values, dtype=numpy.uint32)` is your own conversion, not an API fallback; from a Python list it allocates an aligned, C-contiguous array. `numpy.ascontiguousarray` alone does not fix an already C-contiguous but misaligned buffer.

The `ends` array from `batch` is `uint64`; it is not decode input. Use it to slice the flat `uint32` ids.

## Compatibility

Python 0.3.0 requires this array contract. Python 0.2.0 accepted Python lists and tuples. Historical 0.2.0 examples, release notes, and benchmark results remain valid for that wheel and are not rewritten. The root README, [API guide](API.md), and [migration guide](MIGRATION.md) describe Python 0.3.0 and Rust 0.2.0, and pass arrays directly. The 0.2.0 release notes and the 0.2 API changes page carry a banner pointing here.

The uint32 decode benchmark is separate from the published 0.2.0 measurements: [docs/BENCHMARK_UINT32_DECODE.md](BENCHMARK_UINT32_DECODE.md). It uses the pre-version-bump development wheel, whose binding implementation is the same; the artifact and version are recorded as measured, and there was no retiming for the release. It does not change the published 0.2.0 package results.
