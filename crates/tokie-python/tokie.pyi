from collections.abc import Sequence
from typing import Optional

import numpy
import numpy.typing

class Tokenizer:
    """One tokenizer. `encode` and `batch` return owner-backed NumPy arrays."""

    @staticmethod
    def open(path: str) -> "Tokenizer":
        """Read a `.tokie` file or a `tokenizer.json`."""
        ...
    @staticmethod
    def json(text: str) -> "Tokenizer":
        """Parse the text of a `tokenizer.json`."""
        ...
    @staticmethod
    def from_pretrained(repo: str) -> "Tokenizer":
        """Download the repo's `.tokie` file, or `tokenizer.json` when the repo has none."""
        ...
    @staticmethod
    def from_json(path: str) -> "Tokenizer":
        """Read a `.tokie` file or a `tokenizer.json` from a path. Alias for `open`."""
        ...
    @staticmethod
    def from_file(path: str) -> "Tokenizer":
        """Read a `.tokie` file or a `tokenizer.json` from a path. Alias for `open`."""
        ...
    def encode(self, text: str) -> numpy.typing.NDArray[numpy.uint32]:
        """Append-style encode. Specials in `text` are emitted; the template stays off."""
        ...
    def encode_plus(
        self,
        text: str,
        pair: Optional[str] = None,
        max_len: Optional[int] = None,
        truncate: bool = False,
        pad: bool = False,
    ) -> numpy.typing.NDArray[numpy.uint32]:
        """`encode`, then the template. `pair` uses the pair template.
        `truncate` cuts to `max_len`. `pad` extends to `max_len`."""
        ...
    def count_tokens(self, text: str) -> int:
        """How many ids `encode` would return for `text`. The template stays off."""
        ...
    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """One length per text, from `batch`. The template stays off."""
        ...
    def decode(
        self,
        ids: numpy.typing.NDArray[numpy.uint32],
        skip: bool = False,
    ) -> str:
        """Text for `ids`. `ids` must be a 1-D `uint32` NumPy array that is
        C-contiguous, aligned, and native-endian. Empty and read-only arrays are
        accepted. A wrong dtype, non-native byte order, or other rank raises
        `TypeError`; a non-empty non-contiguous or misaligned array raises
        `ValueError`. Do not mutate `ids` while this call runs. `skip` omits
        specials. Invalid UTF-8 is replaced."""
        ...
    def decode_bytes(
        self,
        ids: numpy.typing.NDArray[numpy.uint32],
        skip: bool = False,
    ) -> bytes:
        """Raw bytes for `ids`. `ids` must be a 1-D `uint32` NumPy array that is
        C-contiguous, aligned, and native-endian. Empty and read-only arrays are
        accepted. A wrong dtype, non-native byte order, or other rank raises
        `TypeError`; a non-empty non-contiguous or misaligned array raises
        `ValueError`. Do not mutate `ids` while this call runs. `skip` omits
        specials. Invalid UTF-8 stays as bytes."""
        ...
    def batch(
        self, texts: Sequence[str]
    ) -> tuple[
        numpy.typing.NDArray[numpy.uint32], numpy.typing.NDArray[numpy.uint64]
    ]:
        """Encode each text with `encode`. Returns `(ids, ends)` as `uint32` and
        `uint64` arrays. `ends[i]` is the id count after text `i`."""
        ...
    def threads(self, n: int) -> None:
        """How many workers `batch` uses."""
        ...
    def token(self, id: int) -> Optional[str]:
        """The vocab string for `id`."""
        ...
    def id(self, token: str) -> Optional[int]:
        """The id for a vocab string."""
        ...
    def save(self, path: str) -> None:
        """Write this tokenizer as a `.tokie` file."""
        ...
    @property
    def vocab_size(self) -> int:
        """The largest id plus one, counting empty slots."""
        ...
    @property
    def pad_token_id(self) -> Optional[int]:
        """The pad id, when the vocab spells it `[PAD]` or `<pad>`."""
        ...
    def get_vocab(self) -> dict[str, int]:
        """The vocabulary with its raw `tokenizer.json` spelling."""
        ...
    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        """Ids the template adds for one text, or for a pair."""
        ...
    def __repr__(self) -> str: ...

class TokieError(Exception):
    """Error raised by tokie operations."""
    ...
