"""The 0.2 calls. encode leaves the template off. encode_plus applies it."""

import tokie

tokenizer = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
plain = tokenizer.encode("Hello world")
wrapped = tokenizer.encode_plus("Hello world")
print(plain.tolist())
print(wrapped.tolist())
