"""Synthetic checks for the mainline tokenizer API additions.

Every fixture is built in the test. Nothing loads a Hub cache file, so these
tests do not skip.
"""

import ast
import inspect
import json
from pathlib import Path

import numpy
import pytest

import tokie

STUB = Path(__file__).resolve().parents[1] / "tokie.pyi"

# A BPE vocab with holes, a raw non-ASCII spelling pair, a `[PAD]` spelling,
# one added special, and both template runs.
FULL = {
    "model": {
        "type": "BPE",
        "ignore_merges": True,
        # Ã and Ã© are the byte-level spellings of the bytes c3 and c3 a9.
        "vocab": {
            "a": 0,
            "b": 1,
            "\u00c3": 2,
            "\u00c3\u00a9": 3,
            "[PAD]": 9,
            "ab": 10,
        },
        "merges": ["a b"],
    },
    "added_tokens": [{"id": 4, "content": "[MASK]", "special": True}],
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [5], "tokens": ["[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [6], "tokens": ["[SEP]"]},
        },
    },
}
FULL_TEXT = json.dumps(FULL)

# No template and no pad spelling. `[pad]` must not count as a pad token.
NO_PAD = {
    "model": {
        "type": "BPE",
        "ignore_merges": True,
        "vocab": {"a": 0, "b": 1, "[pad]": 2, "ab": 3},
        "merges": ["a b"],
    }
}
NO_PAD_TEXT = json.dumps(NO_PAD)

ANGLE_PAD = {
    "model": {
        "type": "BPE",
        "ignore_merges": True,
        "vocab": {"a": 0, "b": 1, "<pad>": 2, "ab": 3},
        "merges": ["a b"],
    }
}
ANGLE_PAD_TEXT = json.dumps(ANGLE_PAD)


@pytest.fixture(scope="module")
def full() -> tokie.Tokenizer:
    return tokie.Tokenizer.json(FULL_TEXT)


def test_from_json_and_from_file_are_path_aliases(tmp_path):
    json_path = tmp_path / "tokenizer.json"
    json_path.write_text(FULL_TEXT, encoding="utf-8")
    # A name that says nothing about the contents still loads by content.
    renamed = tmp_path / "model.bin"
    renamed.write_text(FULL_TEXT, encoding="utf-8")

    from_json = tokie.Tokenizer.from_json(str(json_path))
    from_file = tokie.Tokenizer.from_file(str(renamed))
    opened = tokie.Tokenizer.open(str(json_path))

    assert from_json.vocab_size == from_file.vocab_size == opened.vocab_size == 11
    for tok in (from_json, from_file):
        assert tok.encode("a").tolist() == opened.encode("a").tolist()
        assert tok.decode(tok.encode("a").tolist()) == "a"

    # The string is the json text, not a filename, and must raise.
    with pytest.raises(tokie.TokieError):
        tokie.Tokenizer.from_json(FULL_TEXT)
    with pytest.raises(tokie.TokieError):
        tokie.Tokenizer.from_file(FULL_TEXT)


def test_get_vocab_uses_raw_spellings(full):
    vocab = full.get_vocab()
    assert isinstance(vocab, dict)
    assert vocab == {
        "a": 0,
        "b": 1,
        "\u00c3": 2,
        "\u00c3\u00a9": 3,
        "[PAD]": 9,
        "[MASK]": 4,
        "ab": 10,
    }
    # The raw spelling is not the decoded text.
    assert "\u00e9" not in vocab
    assert full.token(3) == "\u00c3\u00a9"
    assert full.decode([3]) == "\u00e9"
    assert all(isinstance(key, str) for key in vocab)
    assert all(type(value) is int for value in vocab.values())


def test_vocab_size_counts_sparse_slots(full):
    assert full.vocab_size == 11  # ids 0..=10, with 5..=8 empty
    assert len(full.get_vocab()) == 7
    assert full.vocab_size != len(full.get_vocab())


def test_pad_token_id_only_accepts_pad_and_angle_pad(full):
    assert full.pad_token_id == 9
    assert tokie.Tokenizer.json(NO_PAD_TEXT).pad_token_id is None
    assert tokie.Tokenizer.json(ANGLE_PAD_TEXT).pad_token_id == 2


def test_num_special_tokens_to_add_single_and_pair(full):
    assert full.num_special_tokens_to_add() == 2
    assert full.num_special_tokens_to_add(False) == 2
    assert full.num_special_tokens_to_add(True) == 3

    single = full.encode("a").tolist()
    plus = full.encode_plus("a").tolist()
    pair = full.encode_plus("a", pair="b").tolist()
    assert len(plus) - len(single) == 2
    assert len(pair) - len(single) - len(full.encode("b").tolist()) == 3


def test_num_special_tokens_to_add_without_template():
    lean = tokie.Tokenizer.json(NO_PAD_TEXT)
    assert lean.num_special_tokens_to_add() == 0
    assert lean.num_special_tokens_to_add(True) == 0


def test_decode_bytes_keeps_invalid_utf8(full):
    assert isinstance(full.decode_bytes([2]), bytes)
    assert full.decode_bytes([2]) == b"\xc3"
    # decode is the lossy view of the same bytes.
    assert full.decode([2]) == "\ufffd"
    assert full.decode_bytes([3]) == "\u00e9".encode()

    kept = full.decode_bytes([2, 4, 3])
    assert kept == b"\xc3[MASK]\xc3\xa9"
    # [MASK] is an added special, so skip drops it and keeps the raw bytes.
    assert full.decode_bytes([2, 4, 3], skip=True) == b"\xc3\xc3\xa9"
    assert full.decode_bytes([2, 4, 3], True) == b"\xc3\xc3\xa9"


def test_decode_bytes_unknown_id_raises(full):
    with pytest.raises(tokie.TokieError):
        full.decode_bytes([99])
    with pytest.raises(tokie.TokieError):
        full.decode([99])


def test_decode_accepts_encoded_array_and_sequences(full):
    ids = full.encode("a")
    assert full.decode(ids) == "a"
    assert full.decode(ids.tolist()) == "a"
    assert full.decode(tuple(ids.tolist())) == "a"

    assert full.decode_bytes(ids) == b"a"
    assert full.decode_bytes(ids.tolist()) == b"a"
    assert full.decode_bytes(tuple(ids.tolist())) == b"a"


def test_batch_and_count_accept_sequences(full):
    listed = ["a", "b"]
    tupled = ("a", "b")

    ids_list, ends_list = full.batch(listed)
    ids_tuple, ends_tuple = full.batch(tupled)
    assert ids_list.tolist() == ids_tuple.tolist()
    assert ends_list.tolist() == ends_tuple.tolist()
    assert full.count_tokens_batch(listed) == full.count_tokens_batch(tupled)


def test_returned_arrays_remain_stable(full):
    first = full.encode("a")
    first_ids = first.tolist()
    # The array views a buffer kept alive by its base object.
    assert first.base is not None

    batch_ids, batch_ends = full.batch(["b", "ab"])
    batch_flat = batch_ids.tolist()
    batch_ends_list = batch_ends.tolist()

    for text in ("b", "ab", ""):
        full.encode(text)
        full.encode_plus(text)
    full.decode(first)
    full.decode_bytes(first)

    assert first.tolist() == first_ids
    assert batch_ids.tolist() == batch_flat
    assert batch_ends.tolist() == batch_ends_list


def test_encode_and_batch_fast_paths_unchanged(full):
    texts = ["a", "b", "ab", ""]

    one = full.encode("ab")
    assert one.dtype == numpy.uint32
    assert one.ndim == 1

    ids, ends = full.batch(texts)
    assert ids.dtype == numpy.uint32
    assert ends.dtype == numpy.uint64

    flat = ids.tolist()
    ends_list = ends.tolist()
    expected = []
    cursor = 0
    for text in texts:
        at = full.encode(text).tolist()
        expected.append(cursor + len(at))
        assert flat[cursor : expected[-1]] == at
        cursor = expected[-1]
    assert ends_list == expected
    assert cursor == len(flat)


def test_no_compat_namespace_or_duplicate_class():
    assert not hasattr(tokie, "Encoding")
    assert not hasattr(tokie, "compat")

    named = [
        name
        for name, value in vars(tokie).items()
        if isinstance(value, type) and value.__name__ == "Tokenizer"
    ]
    assert named == ["Tokenizer"]


def _stub_class():
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    tokenizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Tokenizer"
    )
    return tokenizer, classes


def _stub_parts():
    tokenizer, classes = _stub_class()
    methods, properties, statics = set(), set(), set()
    for node in tokenizer.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        methods.add(node.name)
        decorators = {
            item.id
            for item in node.decorator_list
            if isinstance(item, ast.Name)
        }
        if "property" in decorators:
            properties.add(node.name)
        if "staticmethod" in decorators:
            statics.add(node.name)
    return methods, properties, statics, classes


def test_stub_matches_runtime_api():
    methods, properties, statics, classes = _stub_parts()
    runtime = {name for name in dir(tokie.Tokenizer) if not name.startswith("_")}
    runtime.add("__repr__")

    assert methods == runtime
    assert classes == ["Tokenizer", "TokieError"]
    assert "Encoding" not in classes

    assert properties == {"vocab_size", "pad_token_id"}
    for name in properties:
        assert not callable(getattr(tokie.Tokenizer, name))
    assert statics == {"open", "json", "from_pretrained", "from_json", "from_file"}

    for name in methods - {"__repr__"}:
        assert hasattr(tokie.Tokenizer, name), name


def test_stub_defaults_match_runtime():
    _, properties, _, _ = _stub_parts()
    tokenizer, _ = _stub_class()
    for node in tokenizer.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "__repr__" or node.name in properties:
            continue
        args = node.args.args
        names = [arg.arg for arg in args]
        if names and names[0] == "self":
            names = names[1:]
        stub_defaults = {}
        if node.args.defaults:
            for arg, value in zip(args[-len(node.args.defaults) :], node.args.defaults):
                stub_defaults[arg.arg] = ast.literal_eval(value)

        params = inspect.signature(getattr(tokie.Tokenizer, node.name)).parameters
        runtime_names = list(params)
        if runtime_names and runtime_names[0] == "self":
            runtime_names = runtime_names[1:]
        runtime_defaults = {
            name: param.default
            for name, param in params.items()
            if param.default is not inspect.Parameter.empty
        }
        assert runtime_names == names, node.name
        assert runtime_defaults == stub_defaults, node.name
