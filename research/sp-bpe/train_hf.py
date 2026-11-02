"""Train a SentencePiece-BPE-shaped tokenizer (Metaspace + BPE) and export
the artifacts the Rust harness needs: explicit vocab + ordered merges."""
import json, sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
OUT = sys.argv[2] if len(sys.argv) > 2 else "spbpe_hf32k"

tok = Tokenizer(models.BPE(unk_token="<unk>", fuse_unk=True, byte_fallback=True))
# Metaspace = the SentencePiece convention: replace ' ' with U+2581, prepend on first word.
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=["<unk>", "<s>", "</s>"],
    initial_alphabet=[],
    show_progress=True,
)
tok.train(["corpus/clean.txt"], trainer)
tok.save(f"{OUT}.json")

data = json.load(open(f"{OUT}.json"))
model = data["model"]
vocab, merges = model["vocab"], model["merges"]
print(f"TRAINED vocab={len(vocab)} merges={len(merges)}", flush=True)

# Export flat artifacts for the dependency-free Rust harness.
# vocab.tsv : <id>\t<hex-encoded piece bytes>
inv = sorted(vocab.items(), key=lambda kv: kv[1])
with open(f"{OUT}.vocab.tsv", "w") as f:
    for piece, idx in inv:
        f.write(f"{idx}\t{piece.encode('utf-8').hex()}\n")

# merges.tsv : <left_id>\t<right_id>  in rank order
with open(f"{OUT}.merges.tsv", "w") as f:
    n_ok = 0
    for m in merges:
        left, right = (m if isinstance(m, (list, tuple)) else m.split(" ", 1))
        if left in vocab and right in vocab:
            f.write(f"{vocab[left]}\t{vocab[right]}\n")
            n_ok += 1
print(f"EXPORTED merges_resolved={n_ok}", flush=True)
