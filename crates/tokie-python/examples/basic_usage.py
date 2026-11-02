"""Load a tokenizer.json, encode, and decode."""

import tokie

tokenizer = tokie.Tokenizer.open("tokenizer.json")
ids = tokenizer.encode("Hello world")
plus = tokenizer.encode_plus("Hello world")
print(ids.tolist())
print(plus.tolist())
print(tokenizer.decode(plus.tolist(), skip=True))
print(tokenizer.count_tokens("Hello world"))
print(tokenizer.count_tokens_batch(["Hello", "world"]))
