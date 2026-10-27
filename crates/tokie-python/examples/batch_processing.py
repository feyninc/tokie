"""Encode many texts with Tokenizer.batch."""

import tokie

tokenizer = tokie.Tokenizer.open("tokenizer.json")
tokenizer.threads(4)
ids, ends = tokenizer.batch(["Hello world", "a12b", ""])
print(ids.tolist())
print(ends.tolist())
