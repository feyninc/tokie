"""Side-by-side comparison: HuggingFace tokenizers vs tokie.

Run with: pip install tokenizers tokie
"""

import time

# ---- tokie ----
import tokie

start = time.perf_counter()
tok = tokie.Tokenizer.from_pretrained("bert-base-uncased")
tokie_load = time.perf_counter() - start

start = time.perf_counter()
tokie_enc = tok.encode("The quick brown fox jumps over the lazy dog.")
tokie_time = time.perf_counter() - start

print("=== tokie ===")
print(f"  Load time:  {tokie_load*1000:.1f} ms")
print(f"  Encode time: {tokie_time*1000:.3f} ms")
print(f"  Token IDs:  {tokie_enc.ids}")
print(f"  Attn mask:  {tokie_enc.attention_mask}")
print(f"  Type IDs:   {tokie_enc.type_ids}")

# ---- HuggingFace ----
try:
    from tokenizers import Tokenizer as HfTokenizer

    start = time.perf_counter()
    hf = HfTokenizer.from_pretrained("bert-base-uncased")
    hf_load = time.perf_counter() - start

    start = time.perf_counter()
    hf_enc = hf.encode("The quick brown fox jumps over the lazy dog.")
    hf_time = time.perf_counter() - start

    print("\n=== HuggingFace ===")
    print(f"  Load time:  {hf_load*1000:.1f} ms")
    print(f"  Encode time: {hf_time*1000:.3f} ms")
    print(f"  Token IDs:  {hf_enc.ids}")
    print(f"  Attn mask:  {hf_enc.attention_mask}")
    print(f"  Type IDs:   {hf_enc.type_ids}")

    # Verify identical output
    print("\n=== Comparison ===")
    print(f"  IDs match:      {tokie_enc.ids == hf_enc.ids}")
    print(f"  Attn match:     {tokie_enc.attention_mask == hf_enc.attention_mask}")
    print(f"  Type IDs match: {tokie_enc.type_ids == hf_enc.type_ids}")
    print(f"  Load speedup:   {hf_load/tokie_load:.1f}x")

except ImportError:
    print("\n(Install `tokenizers` to compare: pip install tokenizers)")
