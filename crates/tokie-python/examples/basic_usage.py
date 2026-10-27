"""Load a tokenizer.json, encode, and decode."""

import tokie

tokenizer = tokie.Tokenizer.open("tokenizer.json")
ids = tokenizer.encode("Hello world")
print(ids.tolist())
print(tokenizer.decode(ids.tolist()))
