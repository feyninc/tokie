from typing import Optional

import numpy

class Encoding:
    """Result of encoding text, with token IDs, attention mask, and type IDs."""

    ids: list[int]
    tokens: list[str]
    special_tokens_mask: list[int]
    attention_mask: list[int]
    type_ids: list[int]
    offsets: list[tuple[int, int]]
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Tokenizer:
    """Fast, correct tokenizer. Supports BPE, WordPiece, and Unigram."""

    @staticmethod
    def from_json(path: str) -> "Tokenizer":
        """Load a tokenizer from a HuggingFace tokenizer.json file."""
        ...
    @staticmethod
    def from_file(path: str) -> "Tokenizer":
        """Load a tokenizer from a .tkz binary file."""
        ...
    @staticmethod
    def from_pretrained(repo_id: str) -> "Tokenizer":
        """Download and load a tokenizer from the HuggingFace Hub."""
        ...
    def __call__(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode text or a text pair. Usage: tokenizer("text") or tokenizer("text_a", "text_b")."""
        ...
    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        """Encode text into an Encoding (ids, attention_mask, type_ids)."""
        ...
    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[Encoding]:
        """Encode multiple texts in parallel, returning a list of Encoding objects."""
        ...
    def encode_pair(
        self, text_a: str, text_b: str, add_special_tokens: bool = True
    ) -> Encoding:
        """Encode a pair of texts (e.g. for cross-encoder models)."""
        ...
    def encode_with_offsets(
        self, text: str, add_special_tokens: bool = True
    ) -> Encoding:
        """Encode text into an Encoding with byte offsets into the (normalized) input."""
        ...
    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode raw bytes into token IDs."""
        ...
    def decode(self, tokens: list[int]) -> Optional[str]:
        """Decode token IDs back to a string. Returns None if not valid UTF-8."""
        ...
    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode token IDs back to raw bytes."""
        ...
    def decode_batch(self, sequences: list[list[int]]) -> list[Optional[str]]:
        """Decode multiple token ID sequences."""
        ...
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert a token ID to its string representation."""
        ...
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        ...
    def get_vocab(self) -> dict[str, int]:
        """Get the full vocabulary as a dict mapping token strings to IDs."""
        ...
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        ...
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts in parallel."""
        ...
    def encode_files(
        self,
        paths: list[str],
        separator: bytes = b"<|endoftext|>",
        add_special_tokens: bool = False,
    ) -> tuple["numpy.ndarray", "numpy.ndarray"]:
        """Encode corpus files entirely in Rust: read bytes, split documents on
        the separator, drop empty documents, encode in parallel. Returns
        (ids, offsets) numpy arrays — uint32 concatenated ids and uint64
        document boundaries of length ndocs + 1; document i is
        ids[offsets[i]:offsets[i + 1]]. Invalid UTF-8 is lossy-converted."""
        ...
    def count_tokens_files(
        self, paths: list[str], separator: bytes = b"<|endoftext|>"
    ) -> int:
        """Total token count across corpus files, split and encoded as in
        encode_files, without materializing ids."""
        ...
    def save(self, path: str) -> None:
        """Save the tokenizer to a .tkz binary file."""
        ...
    def enable_padding(
        self,
        *,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        """Enable padding for encode_batch."""
        ...
    def enable_truncation(
        self,
        max_length: int,
        *,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None:
        """Enable truncation."""
        ...
    def no_padding(self) -> None:
        """Disable padding."""
        ...
    def no_truncation(self) -> None:
        """Disable truncation."""
        ...
    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        ...
    @property
    def pad_token_id(self) -> Optional[int]:
        """The pad token ID, if set."""
        ...
    @property
    def padding(self) -> Optional[dict[str, object]]:
        """Get current padding configuration, or None if disabled."""
        ...
    @property
    def truncation(self) -> Optional[dict[str, object]]:
        """Get current truncation configuration, or None if disabled."""
        ...
    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        """Number of special tokens added for a single sequence or pair."""
        ...
    def __repr__(self) -> str: ...

class TokieError(Exception):
    """Error raised by tokie operations."""
    ...
