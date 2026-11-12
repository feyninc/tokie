"""Input contract for `decode` and `decode_bytes`.

Both methods take one 1-D native `uint32` NumPy array. The array must be
C-contiguous and aligned when non-empty. Lists, tuples, scalars, other dtypes,
other ranks, non-native byte order, and non-empty non-contiguous or misaligned
arrays are rejected. Empty and read-only arrays are accepted. No test needs an
external tokenizer: the fixture is a small synthetic BPE json built in this
file.
"""

import json

import numpy
import pytest

import tokie

# `ab` sits at id 5, so ids 2..=4 are a gap the vocab never fills.
JSON = json.dumps(
    {
        "model": {
            "type": "BPE",
            "ignore_merges": True,
            "vocab": {"a": 0, "b": 1, "ab": 5},
            "merges": ["a b"],
        }
    }
)

MAX_U32 = 2**32 - 1
NON_NATIVE = numpy.dtype(">u4" if numpy.little_endian else "<u4")
METHODS = ["decode", "decode_bytes"]


@pytest.fixture(scope="module")
def full() -> tokie.Tokenizer:
    return tokie.Tokenizer.json(JSON)


def call(tok, method, ids):
    return getattr(tok, method)(ids)


def expected(method, text):
    return text if method == "decode" else text.encode()


def test_accepts_array_from_encode(full):
    ids = full.encode("ab")
    assert ids.dtype == numpy.uint32
    assert ids.ndim == 1
    for method in METHODS:
        assert call(full, method, ids) == expected(method, "ab")


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param(numpy.array([], dtype=numpy.uint32), id="plain"),
        pytest.param(numpy.zeros(0, dtype=numpy.uint32)[::2], id="zero-stride"),
        pytest.param(
            numpy.frombuffer(bytearray(1), dtype=numpy.uint32, offset=1),
            id="misaligned-offset",
        ),
    ],
)
def test_accepts_empty_array(full, empty):
    assert empty.dtype == numpy.uint32
    assert empty.ndim == 1
    assert empty.size == 0
    for method in METHODS:
        assert call(full, method, empty) == expected(method, "")


def test_accepts_readonly_array(full):
    ids = full.encode("ab")
    ids.setflags(write=False)
    for method in METHODS:
        assert call(full, method, ids) == expected(method, "ab")


def test_accepts_contiguous_slice(full):
    base = numpy.array([0, 1, 0], dtype=numpy.uint32)
    view = base[1:2]
    assert view.flags.c_contiguous
    for method in METHODS:
        assert call(full, method, view) == expected(method, "b")


def test_accepts_batch_flat_array(full):
    ids, ends = full.batch(["a", "b"])
    assert ids.dtype == numpy.uint32
    assert int(ends[-1]) == len(ids)
    for method in METHODS:
        assert call(full, method, ids) == expected(method, "ab")


def test_accepts_ndarray_subclass(full):
    class Sub(numpy.ndarray):
        pass

    ids = numpy.array([0], dtype=numpy.uint32).view(Sub)
    for method in METHODS:
        assert call(full, method, ids) == expected(method, "a")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("id", [2, 99, MAX_U32], ids=["gap", "unknown", "max-u32"])
def test_core_errors_still_raise_tokie_error(full, method, id):
    ids = numpy.array([id], dtype=numpy.uint32)
    with pytest.raises(tokie.TokieError):
        getattr(full, method)(ids)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param([0], id="list"),
        pytest.param((0,), id="tuple"),
        pytest.param(0, id="int"),
        pytest.param(b"\x00\x00\x00\x00", id="bytes"),
        pytest.param(memoryview(b"\x00\x00\x00\x00"), id="memoryview"),
    ],
)
def test_rejects_non_array_input(full, method, value):
    with pytest.raises(TypeError):
        getattr(full, method)(value)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(numpy.int64, id="int64"),
        pytest.param(numpy.uint64, id="uint64"),
        pytest.param(numpy.float64, id="float64"),
        pytest.param(numpy.bool_, id="bool"),
        pytest.param(object, id="object"),
    ],
)
def test_rejects_other_dtypes(full, method, dtype):
    ids = numpy.array([0], dtype=dtype)
    with pytest.raises(TypeError):
        getattr(full, method)(ids)


@pytest.mark.parametrize("method", METHODS)
def test_rejects_non_native_byte_order(full, method):
    ids = numpy.array([0], dtype=NON_NATIVE)
    assert ids.dtype == NON_NATIVE
    with pytest.raises(TypeError) as excinfo:
        getattr(full, method)(ids)
    assert type(excinfo.value) is TypeError


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "ids",
    [
        pytest.param(numpy.array(0, dtype=numpy.uint32), id="rank-0"),
        pytest.param(numpy.array([[0, 1]], dtype=numpy.uint32), id="rank-2"),
    ],
)
def test_rejects_other_ranks(full, method, ids):
    with pytest.raises(TypeError):
        getattr(full, method)(ids)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "ids",
    [
        pytest.param(
            numpy.array([0, 1, 0, 1], dtype=numpy.uint32)[::2],
            id="strided",
        ),
        pytest.param(
            numpy.array([0, 1, 0, 1], dtype=numpy.uint32)[::-1],
            id="negative-stride",
        ),
    ],
)
def test_rejects_noncontiguous_arrays(full, method, ids):
    assert not ids.flags.c_contiguous
    with pytest.raises(ValueError):
        getattr(full, method)(ids)


@pytest.mark.parametrize("method", METHODS)
def test_rejects_misaligned_array(full, method):
    ids = numpy.frombuffer(bytearray(4 * 3 + 1), dtype=numpy.uint32, offset=1)
    assert ids.dtype == numpy.uint32
    assert ids.size > 0
    assert not ids.flags.aligned
    with pytest.raises(ValueError):
        getattr(full, method)(ids)
