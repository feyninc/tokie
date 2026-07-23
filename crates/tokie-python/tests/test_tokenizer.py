import pytest
import tokie


def test_from_pretrained_bert():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.vocab_size == 30522
    assert repr(t) == "Tokenizer(vocab_size=30522)"


def test_encode_returns_encoding():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("Hello, world!")
    assert isinstance(enc, tokie.Encoding)
    assert isinstance(enc.ids, list)
    assert isinstance(enc.attention_mask, list)
    assert isinstance(enc.type_ids, list)
    assert len(enc) == len(enc.ids)
    assert all(m == 1 for m in enc.attention_mask)
    assert all(t == 0 for t in enc.type_ids)


def test_encode_decode_roundtrip():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    text = "Hello, world!"
    enc = t.encode(text)
    decoded = t.decode(enc.ids)
    assert isinstance(enc.ids, list)
    assert all(isinstance(tok, int) for tok in enc.ids)
    assert isinstance(decoded, str)


def test_encode_without_special_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    with_special = t.encode("hello", add_special_tokens=True)
    without_special = t.encode("hello", add_special_tokens=False)
    # With special tokens should be longer ([CLS] + tokens + [SEP])
    assert len(with_special) == len(without_special) + 2


def test_count_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    count = t.count_tokens("Hello, world!")
    tokens = t.encode("Hello, world!", add_special_tokens=False)
    assert count == len(tokens)


def test_encode_pair():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    pair = t.encode_pair("How are you?", "I am fine.")
    assert isinstance(pair, tokie.Encoding)
    assert isinstance(pair.ids, list)
    assert isinstance(pair.attention_mask, list)
    assert isinstance(pair.type_ids, list)
    assert len(pair) == len(pair.ids)
    assert len(pair.attention_mask) == len(pair.ids)
    assert len(pair.type_ids) == len(pair.ids)
    # type_ids should have 0s for first seq, 1s for second
    assert 0 in pair.type_ids
    assert 1 in pair.type_ids


def test_encode_bytes():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    tokens = t.encode_bytes(b"hello")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_decode_bytes():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("hello")
    raw = t.decode_bytes(enc.ids)
    assert isinstance(raw, bytes)


def test_save_load_roundtrip(tmp_path):
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    path = str(tmp_path / "test.tkz")
    t.save(path)
    t2 = tokie.Tokenizer.from_file(path)
    assert t2.vocab_size == t.vocab_size
    assert t2.encode("test").ids == t.encode("test").ids


def test_error_handling():
    with pytest.raises(tokie.TokieError):
        tokie.Tokenizer.from_file("/nonexistent.tkz")


def test_gpt2():
    t = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
    enc = t.encode("Hello, world!", add_special_tokens=False)
    assert isinstance(enc.ids, list)
    assert len(enc) > 0
    decoded = t.decode(enc.ids)
    assert decoded == "Hello, world!"


def test_encode_batch_basic():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["Hello world", "How are you?", "This is a test", "Goodbye", "One more"]
    batch = t.encode_batch(texts)
    assert len(batch) == 5
    for i, text in enumerate(texts):
        assert batch[i].ids == t.encode(text).ids


def test_encode_batch_empty():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.encode_batch([]) == []


def test_encode_batch_preserves_order():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = [f"sentence number {i} with some content" for i in range(50)]
    batch = t.encode_batch(texts)
    assert len(batch) == 50
    for i, text in enumerate(texts):
        assert batch[i].ids == t.encode(text).ids, f"Mismatch at index {i}"


def test_encode_batch_without_special_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["hello", "world"]
    with_special = t.encode_batch(texts, add_special_tokens=True)
    without_special = t.encode_batch(texts, add_special_tokens=False)
    for ws, wos in zip(with_special, without_special):
        assert len(ws) == len(wos) + 2  # [CLS] + tokens + [SEP]


def test_count_tokens_batch():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["Hello world", "How are you?", "Test"]
    counts = t.count_tokens_batch(texts)
    assert len(counts) == 3
    for i, text in enumerate(texts):
        assert counts[i] == t.count_tokens(text)


def test_count_tokens_batch_empty():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.count_tokens_batch([]) == []


def test_defaults_no_padding_no_truncation():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("Hello world")
    # No padding by default — attention_mask all 1s, type_ids all 0s
    assert all(m == 1 for m in enc.attention_mask)
    assert all(t == 0 for t in enc.type_ids)


def test_pad_token_id_property():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    # BERT should have [PAD] token
    assert t.pad_token_id is not None
    assert t.pad_token_id == 0  # [PAD] is typically token 0 in BERT


def test_enable_truncation():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    t.enable_truncation(max_length=8)
    enc = t.encode("This is a test sentence that should be truncated to fit", add_special_tokens=True)
    assert len(enc) <= 8


def test_enable_padding_batch():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    t.enable_padding()
    results = t.encode_batch(["Hello world", "Short", "A much longer sentence for testing purposes"])
    # All should be same length (padded to longest)
    lengths = [len(r) for r in results]
    assert len(set(lengths)) == 1  # all same length
    # Shorter sequences should have 0s in attention_mask
    max_len = lengths[0]
    for r in results:
        assert len(r.attention_mask) == max_len
        # Check that non-padded tokens have attention_mask=1
        assert r.attention_mask[0] == 1


def test_enable_padding_fixed():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    t.enable_padding(length=16)
    results = t.encode_batch(["Hello", "World"])
    assert all(len(r) == 16 for r in results)


def test_no_padding_no_truncation():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    t.enable_padding(length=16)
    t.enable_truncation(max_length=8)
    t.no_padding()
    t.no_truncation()
    # Should behave as default now
    enc = t.encode("Hello world test sentence")
    assert all(m == 1 for m in enc.attention_mask)


def test_id_to_token():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    # BERT token 101 = [CLS], 102 = [SEP]
    assert t.id_to_token(101) == "[CLS]"
    assert t.id_to_token(102) == "[SEP]"
    assert t.id_to_token(999999) is None


def test_token_to_id():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.token_to_id("[CLS]") == 101
    assert t.token_to_id("[SEP]") == 102
    assert t.token_to_id("nonexistent_xyz") is None


def test_get_vocab():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    vocab = t.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) > 0
    assert vocab["[CLS]"] == 101


def test_decode_batch():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc1 = t.encode("Hello", add_special_tokens=False)
    enc2 = t.encode("World", add_special_tokens=False)
    results = t.decode_batch([enc1.ids, enc2.ids])
    assert len(results) == 2
    assert results[0] == "hello"
    assert results[1] == "world"


def test_encode_with_offsets():
    t = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
    enc = t.encode_with_offsets("Hello world", add_special_tokens=False)
    assert isinstance(enc, tokie.Encoding)
    assert len(enc.offsets) == len(enc.ids)
    # Offsets should be contiguous
    for i in range(1, len(enc.offsets)):
        assert enc.offsets[i][0] == enc.offsets[i - 1][1]
    # First offset starts at 0, last ends at text byte length
    assert enc.offsets[0][0] == 0
    assert enc.offsets[-1][1] == len("Hello world".encode("utf-8"))


def test_encode_with_offsets_empty():
    t = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
    enc = t.encode_with_offsets("", add_special_tokens=False)
    assert enc.ids == []
    assert enc.offsets == []


def test_encode_with_offsets_has_attention_mask():
    t = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
    enc = t.encode_with_offsets("test", add_special_tokens=False)
    assert all(m == 1 for m in enc.attention_mask)
    assert len(enc.attention_mask) == len(enc.ids)


def test_num_special_tokens_to_add():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.num_special_tokens_to_add(False) == 2  # [CLS] + [SEP]
    assert t.num_special_tokens_to_add(True) == 3  # [CLS] + [SEP] + [SEP]


def test_tokens_property():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("Hello world")
    assert isinstance(enc.tokens, list)
    assert len(enc.tokens) == len(enc.ids)
    # BERT: [CLS] hello world [SEP]
    assert enc.tokens[0] == "[CLS]"
    assert enc.tokens[-1] == "[SEP]"
    assert "hello" in enc.tokens
    assert "world" in enc.tokens


def test_special_tokens_mask():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("Hello world")
    assert isinstance(enc.special_tokens_mask, list)
    assert len(enc.special_tokens_mask) == len(enc.ids)
    # First ([CLS]) and last ([SEP]) should be special
    assert enc.special_tokens_mask[0] == 1
    assert enc.special_tokens_mask[-1] == 1
    # Content tokens should not be special
    for i in range(1, len(enc.special_tokens_mask) - 1):
        assert enc.special_tokens_mask[i] == 0


def test_tokens_without_special():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("hello", add_special_tokens=False)
    assert len(enc.tokens) == len(enc.ids)
    assert all(m == 0 for m in enc.special_tokens_mask)


def test_tokens_encode_pair():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode_pair("Hello", "World")
    assert len(enc.tokens) == len(enc.ids)
    assert len(enc.special_tokens_mask) == len(enc.ids)
    # [CLS] and [SEP] tokens should be marked special
    assert enc.special_tokens_mask[0] == 1  # [CLS]
    assert sum(enc.special_tokens_mask) >= 2  # at least CLS + SEP


def test_tokens_encode_batch():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    batch = t.encode_batch(["Hello", "World"])
    for enc in batch:
        assert len(enc.tokens) == len(enc.ids)
        assert len(enc.special_tokens_mask) == len(enc.ids)
        assert enc.tokens[0] == "[CLS]"


def test_padding_truncation_getters():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.padding is None
    assert t.truncation is None
    t.enable_padding(length=16)
    t.enable_truncation(max_length=8)
    assert t.padding is not None
    assert t.padding["length"] == 16
    assert t.padding["direction"] == "right"
    assert t.truncation is not None
    assert t.truncation["max_length"] == 8
    assert t.truncation["strategy"] == "longest_first"
    t.no_padding()
    t.no_truncation()
    assert t.padding is None
    assert t.truncation is None


def test_call_single():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t("Hello world")
    assert enc.ids == t.encode("Hello world").ids


def test_call_pair():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t("Hello", "World")
    assert enc.ids == t.encode_pair("Hello", "World").ids
    assert 1 in enc.type_ids  # second sequence has type_id 1


def test_call_no_special():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t("hello", add_special_tokens=False)
    assert enc.ids == t.encode("hello", add_special_tokens=False).ids


def test_unigram_t5():
    t = tokie.Tokenizer.from_pretrained("google-t5/t5-small")
    enc = t.encode("Hello world", add_special_tokens=False)
    assert len(enc.ids) > 0
    decoded = t.decode(enc.ids)
    assert decoded == "Hello world"


def test_unigram_xlmr():
    t = tokie.Tokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    enc = t.encode("Hello world", add_special_tokens=False)
    assert len(enc.ids) > 0
    decoded = t.decode(enc.ids)
    assert "Hello world" in decoded


def test_encoding_tokens_and_masks_correct():
    # tokens/special_tokens_mask are computed lazily from ids — they must stay
    # consistent with id_to_token regardless of when they're accessed
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    enc = t.encode("Hello, world!", add_special_tokens=True)
    assert enc.tokens[0] == "[CLS]" and enc.tokens[-1] == "[SEP]"
    assert enc.tokens == [t.id_to_token(i) for i in enc.ids]
    assert enc.special_tokens_mask[0] == 1 and enc.special_tokens_mask[1] == 0
    assert len(enc.special_tokens_mask) == len(enc.ids)
    batch = t.encode_batch(["Hello there", "general Kenobi"], add_special_tokens=False)
    for e in batch:
        assert e.tokens == [t.id_to_token(i) for i in e.ids]
        assert all(m == 0 for m in e.special_tokens_mask)
