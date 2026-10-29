"""Batch encoding with padding and truncation."""

import tokie

tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")

# Enable truncation and padding for fixed-length ML inputs
tokenizer.enable_truncation(max_length=32)
tokenizer.enable_padding(length=32, pad_id=0)

texts = [
    "Short text.",
    "A somewhat longer piece of text for testing purposes.",
    "The quick brown fox jumps over the lazy dog. " * 5,  # will be truncated
]

# encode_batch runs in parallel across all CPU cores
results = tokenizer.encode_batch(texts)

for i, enc in enumerate(results):
    real_tokens = sum(enc.attention_mask)
    print(f"Text {i}: {len(enc)} tokens total, {real_tokens} real, {len(enc) - real_tokens} padding")
    assert len(enc) == 32  # all exactly 32 tokens

# Count tokens in batch (even faster — no attention mask overhead)
counts = tokenizer.count_tokens_batch([t for t in texts])
print(f"\nRaw token counts (before truncation): {counts}")
