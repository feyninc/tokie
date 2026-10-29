"""Basic tokie usage: load, encode, decode."""

import tokie

# Load any HuggingFace tokenizer
tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")
print(f"Vocab size: {tokenizer.vocab_size}")

# Encode text — returns Encoding with ids, attention_mask, type_ids
encoding = tokenizer.encode("Hello, world!")
print(f"Token IDs: {encoding.ids}")
print(f"Attention mask: {encoding.attention_mask}")
print(f"Type IDs: {encoding.type_ids}")

# Decode back to text
text = tokenizer.decode(encoding.ids)
print(f"Decoded: {text}")

# Count tokens without full encoding
count = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {count}")

# Vocabulary access
print(f"Token 101 = {tokenizer.id_to_token(101)}")
print(f"[SEP] = {tokenizer.token_to_id('[SEP]')}")

# Save as .tkz (10x smaller than tokenizer.json)
tokenizer.save("/tmp/bert.tkz")
fast_tokenizer = tokie.Tokenizer.from_file("/tmp/bert.tkz")
assert fast_tokenizer.encode("Hello, world!").ids == encoding.ids
print("Roundtrip OK!")
