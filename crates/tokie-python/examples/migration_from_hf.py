"""The 0.2 calls. encode leaves the template off. wrap applies it."""

import tokie

tokenizer = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
plain = tokenizer.encode("Hello world")
wrapped = tokenizer.wrap("Hello world")
print(plain.tolist())
print(wrapped.tolist())
